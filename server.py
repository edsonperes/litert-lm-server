import os
import json
import time
import uuid
import asyncio
import logging
import threading
import queue
from typing import Optional, Union

import litert_lm
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litert-lm-server")

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models")
MODEL_FILE = os.environ.get("MODEL_FILE", "gemma-4-E2B-it.litertlm")
MODEL_ID = os.environ.get("MODEL_ID", "gemma-4-E2B-it")
PORT = int(os.environ.get("PORT", 8000))

from fastapi.staticfiles import StaticFiles
import pathlib

app = FastAPI(title="LiteRT-LM Server", version="1.3.0")

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
    content: Union[str, list]

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts)
        return v


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)


def get_model_path() -> str:
    return os.path.join(MODEL_DIR, MODEL_FILE)


def build_prompt(messages: list[ChatMessage]) -> str:
    """Constroi prompt completo com system prompt e historico de conversa."""
    parts = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "system":
            parts.append(f"<start_of_turn>user\n[System Instructions]\n{content}<end_of_turn>")
        elif msg.role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif msg.role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


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


@app.get("/props")
async def props():
    """Endpoint compativel com llama.cpp UI."""
    return {
        "default_generation_settings": {
            "model": MODEL_ID,
            "temperature": 0.7,
            "top_p": 0.95,
            "n_predict": 2048,
            "stream": True,
        },
        "total_slots": 1,
        "chat_template": "",
    }


@app.get("/slots")
async def slots():
    """Endpoint compativel com llama.cpp UI."""
    return [
        {
            "id": 0,
            "state": 0,
            "prompt": "",
            "next_token": {"has_next_token": False},
        }
    ]


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

    prompt = build_prompt(request.messages)
    logger.info(f"Prompt ({len(request.messages)} msgs, {len(prompt)} chars): {prompt[-80:]}...")

    if request.stream:
        async def stream_response():
            first = create_chunk("", request.model)
            first["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(first)}\n\n"

            q = queue.Queue()

            def run_inference_thread():
                try:
                    with engine_lock:
                        with engine.create_conversation() as conv:
                            for chunk_data in conv.send_message_async(prompt):
                                text = ""
                                if isinstance(chunk_data, str):
                                    text = chunk_data
                                elif isinstance(chunk_data, dict):
                                    for item in chunk_data.get("content", []):
                                        if item.get("type") == "text":
                                            text += item["text"]
                                if text:
                                    q.put(text)
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    q.put(None)
                finally:
                    q.put(None)

            thread = threading.Thread(target=run_inference_thread, daemon=True)
            thread.start()

            while True:
                try:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, q.get, True, 120
                    )
                except Exception:
                    break
                if text is None:
                    break
                chunk = create_chunk(text, request.model)
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
                    response = conv.send_message(prompt)
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


# Montar arquivos estaticos do llama.cpp UI por ultimo (catch-all)
STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    logger.info(f"llama.cpp UI montada de {STATIC_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
