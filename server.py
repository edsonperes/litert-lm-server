import os
import json
import time
import uuid
import subprocess
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlc-llm-server")

PORT = int(os.environ.get("PORT", 8000))
MODEL = os.environ.get("MODEL", "RedPajama-INCITE-Chat-3B-v1-q4f16_1")
MLC_DIR = "/mlc-llm"
MLC_CLI = f"{MLC_DIR}/build/mlc_chat_cli"

app = FastAPI(title="MLC-LLM GPU Server", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)


def extract_user_message(messages: list[ChatMessage]) -> str:
    """Extrai apenas a ultima mensagem do usuario para enviar ao CLI."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


def run_inference(messages: list[ChatMessage]) -> str:
    user_msg = extract_user_message(messages)
    logger.info(f"Prompt: {user_msg[:100]}...")

    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{MLC_DIR}/build/"

        # Envia mensagem + /exit para sair do CLI
        cli_input = f"{user_msg}\n/exit\n"

        proc = subprocess.run(
            [MLC_CLI, "--local-id", MODEL, "--device", "mali"],
            input=cli_input,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=MLC_DIR,
            env=env,
        )

        output = proc.stdout
        logger.info(f"Raw output length: {len(output)}")

        # Extrair resposta do bot entre <bot>: e <human>:
        # O formato do CLI eh: <human>: [input] <bot>: [resposta] <human>:
        import re

        # Pegar todas as respostas do bot
        bot_responses = re.findall(r'<bot>:\s*(.*?)(?=<human|/exit|\Z)', output, re.DOTALL)

        if bot_responses:
            # Pegar a primeira resposta real (pular as de system prompts)
            response = bot_responses[0].strip()
            # Limpar caracteres de controle e espaços extras
            response = re.sub(r'\s+', ' ', response).strip()
            if response:
                logger.info(f"Response: {response[:100]}...")
                return response

        # Fallback: tentar extrair qualquer texto apos "Loading finished"
        if "Loading finished" in output:
            after_load = output.split("Loading finished")[-1]
            # Remover linhas de sistema
            lines = after_load.split("\n")
            clean_lines = []
            skip_patterns = [
                "Running system", "System prompts", "<human>:",
                "Use /", "/exit", "/help", "/stats", "/reset",
                "/reload", "Use MLC", "Use model", "Loading",
                "arm_release_ver", "You can use"
            ]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if any(line.startswith(p) or p in line for p in skip_patterns):
                    continue
                if line.startswith("<bot>:"):
                    line = line.replace("<bot>:", "").strip()
                clean_lines.append(line)

            response = " ".join(clean_lines).strip()
            if response:
                return response

        if proc.stderr:
            logger.error(f"CLI stderr: {proc.stderr[:500]}")

        raise RuntimeError("Nenhuma resposta gerada pelo modelo")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Inference timeout (120s)")
    except FileNotFoundError:
        raise RuntimeError(f"mlc_chat_cli not found at {MLC_CLI}")


def create_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content.split()),
            "total_tokens": len(content.split()),
        },
    }


def create_chunk(content: str, model: str, finish_reason=None) -> dict:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if content:
        chunk["choices"][0]["delta"] = {"role": "assistant", "content": content}
    return chunk


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "device": "mali-gpu"}


@app.get("/")
async def root():
    return await health()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 1712534400,
                "owned_by": "mlc-ai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        response_text = run_inference(request.messages)
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        # Retorna resposta vazia em vez de 500 para nao quebrar o OpenWebUI
        response_text = "..."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        response_text = "..."

    if request.stream:
        async def stream_response():
            first = create_chunk("", request.model)
            first["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(first)}\n\n"

            # Enviar resposta em chunks de palavras
            words = response_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                chunk = create_chunk(token, request.model)
                yield f"data: {json.dumps(chunk)}\n\n"

            final = create_chunk("", request.model, finish_reason="stop")
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        return JSONResponse(create_response(response_text, request.model))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
