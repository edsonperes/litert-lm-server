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


def format_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    return "\n".join(parts)


def run_inference(prompt: str) -> str:
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{MLC_DIR}/build/"

        proc = subprocess.run(
            [MLC_CLI, "--local-id", MODEL, "--device", "mali"],
            input=f"{prompt}\n/exit\n",
            capture_output=True,
            text=True,
            timeout=120,
            cwd=MLC_DIR,
            env=env,
        )

        output = proc.stdout.strip()

        # Filtrar linhas de output do CLI (prompts, loading messages, etc)
        lines = output.split("\n")
        response_lines = []
        capture = False
        for line in lines:
            # Pular linhas de sistema do CLI
            if line.startswith("Use /") or line.startswith("Loading") or line.startswith("Running"):
                continue
            if ">>> " in line:
                capture = True
                continue
            if capture and line.strip():
                response_lines.append(line)

        response = "\n".join(response_lines).strip()

        if not response and proc.stderr:
            logger.error(f"CLI stderr: {proc.stderr}")
            raise RuntimeError(f"Inference failed: {proc.stderr[:200]}")

        return response if response else output

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
    prompt = format_prompt(request.messages)

    try:
        response_text = run_inference(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

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
