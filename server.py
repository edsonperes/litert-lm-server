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


@app.get("/")
async def root():
    return await chat_ui()


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 4 E2B - Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{background:#16213e;padding:12px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #0f3460}
header h1{font-size:18px;color:#e94560}
header span{font-size:12px;color:#888;background:#0f3460;padding:3px 8px;border-radius:10px}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:12px 16px;border-radius:16px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word;font-size:14px}
.msg.user{align-self:flex-end;background:#0f3460;color:#fff;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#16213e;color:#e0e0e0;border-bottom-left-radius:4px;border:1px solid #0f3460}
.msg.assistant .cursor{display:inline-block;width:8px;height:16px;background:#e94560;animation:blink .8s infinite;vertical-align:text-bottom;margin-left:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
#input-area{background:#16213e;padding:12px 20px;display:flex;gap:10px;border-top:1px solid #0f3460}
#input{flex:1;padding:12px 16px;border:1px solid #0f3460;border-radius:24px;background:#1a1a2e;color:#fff;font-size:14px;outline:none;resize:none;max-height:120px;font-family:inherit}
#input:focus{border-color:#e94560}
#send{padding:12px 24px;background:#e94560;color:#fff;border:none;border-radius:24px;cursor:pointer;font-size:14px;font-weight:600;transition:background .2s}
#send:hover{background:#c73a52}
#send:disabled{background:#555;cursor:not-allowed}
.typing{color:#888;font-style:italic;font-size:13px;padding:4px 16px}
</style>
</head>
<body>
<header>
<h1>Gemma 4 E2B</h1>
<span>LiteRT-LM / CPU</span>
<span>OrangePi 5</span>
</header>
<div id="chat"></div>
<div id="input-area">
<textarea id="input" rows="1" placeholder="Digite sua mensagem..." autofocus></textarea>
<button id="send" onclick="sendMsg()">Enviar</button>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),btn=document.getElementById('send');
let messages=[];
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,120)+'px'});

function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  d.textContent=text;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
  return d;
}

async function sendMsg(){
  const text=input.value.trim();
  if(!text)return;
  input.value='';input.style.height='auto';
  btn.disabled=true;
  addMsg('user',text);
  messages.push({role:'user',content:text});

  const el=addMsg('assistant','');
  el.innerHTML='<span class="cursor"></span>';
  let full='';

  try{
    const res=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'"""+MODEL_ID+"""',messages:messages,stream:true})
    });
    const reader=res.body.getReader();
    const dec=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');
      buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
        try{
          const j=JSON.parse(line.slice(6));
          const c=j.choices?.[0]?.delta?.content;
          if(c){full+=c;el.textContent=full}
        }catch{}
      }
    }
  }catch(e){full='Erro: '+e.message;el.textContent=full}

  el.innerHTML=el.textContent;
  messages.push({role:'assistant',content:full});
  btn.disabled=false;
  input.focus();
  chat.scrollTop=chat.scrollHeight;
}
</script>
</body>
</html>"""


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
