import os
import json
import time
import uuid
import asyncio
import logging
import threading
from typing import Optional

import litert_lm
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

app = FastAPI(title="LiteRT-LM Server", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine global - inicializado no startup
engine = None
engine_lock = threading.Lock()


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


def init_engine():
    global engine
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise RuntimeError(f"Modelo nao encontrado em {model_path}")
    engine = litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU)
    logger.info("Engine LiteRT-LM inicializado (CPU)")


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


@app.on_event("startup")
async def startup():
    model_path = get_model_path()
    if os.path.exists(model_path):
        try:
            init_engine()
        except Exception as e:
            logger.warning(f"Engine nao inicializado no startup: {e}")
    else:
        logger.info(f"Modelo nao encontrado, aguardando download...")


@app.get("/health")
async def health():
    model_path = get_model_path()
    return {
        "status": "ok" if engine else "waiting_for_model",
        "model": MODEL_ID,
        "model_exists": os.path.exists(model_path),
        "engine_loaded": engine is not None,
        "backend": "cpu",
        "streaming": True,
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
    global engine

    if engine is None:
        model_path = get_model_path()
        if os.path.exists(model_path):
            try:
                init_engine()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Erro ao carregar modelo: {e}")
        else:
            raise HTTPException(status_code=503, detail="Modelo nao disponivel. Aguarde o download.")

    user_msg = extract_last_user_message(request.messages)
    logger.info(f"Prompt: {user_msg[:80]}...")

    if request.stream:
        async def stream_response():
            first = create_chunk("", request.model)
            first["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(first)}\n\n"

            try:
                # Coletar resposta completa via API Python
                with engine_lock:
                    with engine.create_conversation() as conv:
                        full_text = ""
                        for chunk_data in conv.send_message_async(user_msg):
                            if isinstance(chunk_data, str):
                                full_text += chunk_data
                            elif isinstance(chunk_data, dict):
                                for item in chunk_data.get("content", []):
                                    if item.get("type") == "text":
                                        full_text += item["text"]

                if not full_text:
                    full_text = "..."

                # Streaming palavra por palavra com delay
                words = full_text.split(" ")
                for i, word in enumerate(words):
                    token = word if i == 0 else f" {word}"
                    chunk = create_chunk(token, request.model)
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if i % 3 == 0:
                        await asyncio.sleep(0.02)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                chunk = create_chunk("...", request.model)
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
        try:
            with engine_lock:
                with engine.create_conversation() as conv:
                    response = conv.send_message(user_msg)
                    if isinstance(response, str):
                        content = response
                    elif isinstance(response, dict):
                        parts = []
                        for item in response.get("content", []):
                            if item.get("type") == "text":
                                parts.append(item["text"])
                        content = "".join(parts)
                    else:
                        content = str(response)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            content = "..."

        logger.info(f"Response: {content[:80]}...")
        return JSONResponse(create_response(content, request.model))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
