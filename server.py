import os
import json
import time
import uuid
import asyncio
import subprocess
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
MODEL_REPO = os.environ.get("MODEL_REPO", "litert-community/gemma-4-E2B-it-litert-lm")
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "python")
BACKEND_TYPE = os.environ.get("BACKEND_TYPE", "cpu")
MODEL_ID = os.environ.get("MODEL_ID", "gemma-4-E2B-it")

app = FastAPI(title="LiteRT-LM Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)


def get_model_path() -> str:
    return os.path.join(MODEL_DIR, MODEL_FILE)


def init_engine():
    global engine
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise RuntimeError(f"Modelo nao encontrado em {model_path}. Aguarde o download.")

    if INFERENCE_BACKEND == "python":
        try:
            import litert_lm
            backend = litert_lm.Backend.CPU
            if BACKEND_TYPE == "gpu":
                try:
                    backend = litert_lm.Backend.GPU
                except AttributeError:
                    logger.warning("GPU backend nao disponivel na API Python, usando CPU")
                    backend = litert_lm.Backend.CPU
            engine = litert_lm.Engine(model_path=model_path, backend=backend)
            logger.info(f"Engine LiteRT-LM inicializado (Python, backend={BACKEND_TYPE})")
        except Exception as e:
            logger.error(f"Falha ao inicializar engine Python: {e}")
            raise
    else:
        engine = "cli"
        logger.info("Usando backend CLI para inferencia")


def format_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<start_of_turn>user\n[System: {msg.content}]<end_of_turn>")
        elif msg.role == "user":
            parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>")
        elif msg.role == "assistant":
            parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def create_completion_response(content: str, model: str, prompt_tokens: int = 0) -> dict:
    completion_tokens = len(content.split())
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> dict:
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


async def infer_python(prompt: str, request: ChatCompletionRequest):
    import litert_lm

    with engine.create_conversation() as conv:
        if request.stream:
            response_stream = conv.send_message_async(prompt)
            for chunk_data in response_stream:
                if isinstance(chunk_data, str):
                    yield chunk_data
                elif isinstance(chunk_data, dict):
                    for item in chunk_data.get("content", []):
                        if item.get("type") == "text":
                            yield item["text"]
        else:
            response = conv.send_message(prompt)
            if isinstance(response, str):
                yield response
            elif isinstance(response, dict):
                text_parts = []
                for item in response.get("content", []):
                    if item.get("type") == "text":
                        text_parts.append(item["text"])
                yield "".join(text_parts)


async def infer_cli(prompt: str, request: ChatCompletionRequest):
    model_path = get_model_path()
    cmd = [
        "litert-lm", "run",
        "--model_path", model_path,
        "--prompt", prompt,
    ]
    if BACKEND_TYPE == "gpu":
        cmd.extend(["--backend", "gpu"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if request.stream:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                if text.strip():
                    yield text
        else:
            stdout, stderr = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"CLI error: {err}")
            yield output

    except FileNotFoundError:
        raise RuntimeError("litert-lm CLI nao encontrado. Verifique a instalacao.")


@app.on_event("startup")
async def startup():
    model_path = get_model_path()
    if os.path.exists(model_path):
        try:
            init_engine()
        except Exception as e:
            logger.warning(f"Engine nao inicializado no startup: {e}")
    else:
        logger.info(f"Modelo nao encontrado em {model_path}, aguardando download...")


@app.get("/health")
async def health():
    model_path = get_model_path()
    model_exists = os.path.exists(model_path)
    return {
        "status": "ok" if engine else "waiting_for_model",
        "model": MODEL_ID,
        "model_loaded": engine is not None,
        "model_exists": model_exists,
        "backend": INFERENCE_BACKEND,
        "backend_type": BACKEND_TYPE,
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
                raise HTTPException(status_code=503, detail=f"Modelo nao carregado: {e}")
        else:
            raise HTTPException(status_code=503, detail="Modelo nao disponivel. Aguarde o download.")

    prompt = format_prompt(request.messages)
    prompt_tokens = len(prompt.split())

    if INFERENCE_BACKEND == "python":
        infer_fn = infer_python
    else:
        infer_fn = infer_cli

    if request.stream:
        async def stream_response():
            first_chunk = create_chunk("", request.model)
            first_chunk["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(first_chunk)}\n\n"

            async for text in infer_fn(prompt, request):
                chunk = create_chunk(text, request.model)
                yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = create_chunk("", request.model, finish_reason="stop")
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        full_response = []
        async for text in infer_fn(prompt, request):
            full_response.append(text)
        content = "".join(full_response)
        return JSONResponse(create_completion_response(content, request.model, prompt_tokens))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
