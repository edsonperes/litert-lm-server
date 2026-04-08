import os
import json
import time
import uuid
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litert-lm-server")

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models")
MODEL_FILE = os.environ.get("MODEL_FILE", "gemma-4-E2B-it.litertlm")
MODEL_ID = os.environ.get("MODEL_ID", "gemma-4-E2B-it")
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(title="LiteRT-LM Server", version="1.0.0")

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
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)


def get_model_path() -> str:
    return os.path.join(MODEL_DIR, MODEL_FILE)


def extract_last_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


async def run_inference(messages: list[ChatMessage]) -> str:
    user_msg = extract_last_user_message(messages)
    model_path = get_model_path()

    logger.info(f"Prompt: {user_msg[:80]}...")

    cmd = ["litert-lm", "run", model_path, "--backend", "cpu", "--prompt", user_msg]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        output = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            logger.error(f"CLI error: {err[:300]}")
            return "Desculpe, ocorreu um erro ao processar sua mensagem."

        if output:
            logger.info(f"Response: {output[:80]}...")
            return output

        return "..."

    except asyncio.TimeoutError:
        logger.error("Inference timeout (120s)")
        return "Desculpe, o tempo de processamento excedeu o limite."
    except FileNotFoundError:
        logger.error("litert-lm CLI not found")
        return "Erro: litert-lm nao encontrado."


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
    model_path = get_model_path()
    return {
        "status": "ok" if os.path.exists(model_path) else "waiting_for_model",
        "model": MODEL_ID,
        "model_exists": os.path.exists(model_path),
        "backend": "cpu",
    }


@app.get("/")
async def root():
    return await health()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1712534400,
                "owned_by": "litert-community",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Modelo nao disponivel. Aguarde o download.")

    try:
        response_text = await run_inference(request.messages)
    except Exception as e:
        logger.error(f"Error: {e}")
        response_text = "..."

    if request.stream:
        async def stream_response():
            first = create_chunk("", request.model)
            first["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(first)}\n\n"

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
