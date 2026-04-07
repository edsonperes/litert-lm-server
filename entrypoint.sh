#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-mlc-ai/gemma-2b-it-q4f16_1-MLC}"
MODEL_DIR="${MODEL_DIR:-/data/models}"
PORT="${PORT:-8000}"

echo "============================================"
echo "  MLC-LLM Server para OrangePi 5 (GPU)"
echo "============================================"
echo "Modelo: ${MODEL_REPO}"
echo "Porta: ${PORT}"
echo "============================================"

# Verificar GPU OpenCL/Mali
echo "[GPU] Verificando OpenCL/Mali..."
if command -v clinfo &>/dev/null; then
    PLATFORMS=$(clinfo -l 2>/dev/null | grep -c "Platform" || true)
    if [ "${PLATFORMS}" -gt 0 ]; then
        echo "[GPU] OpenCL disponivel - ${PLATFORMS} plataforma(s)"
        clinfo -l 2>/dev/null || true
    else
        echo "[GPU] AVISO: Nenhuma plataforma OpenCL detectada"
        echo "[GPU] Verifique se /lib/firmware/mali_csffw.bin esta montado"
    fi
else
    echo "[GPU] AVISO: clinfo nao encontrado"
fi

mkdir -p "${MODEL_DIR}"

# Download do modelo se nao existir
MODEL_PATH="${MODEL_DIR}/$(basename ${MODEL_REPO})"
if [ ! -d "${MODEL_PATH}" ] || [ -z "$(ls -A "${MODEL_PATH}" 2>/dev/null)" ]; then
    echo "[Download] Modelo nao encontrado em ${MODEL_PATH}"
    echo "[Download] Baixando ${MODEL_REPO} do HuggingFace..."
    echo "[Download] Isso pode levar alguns minutos..."

    python3 -c "
from huggingface_hub import snapshot_download
import os

repo_id = os.environ.get('MODEL_REPO', 'mlc-ai/gemma-2b-it-q4f16_1-MLC')
model_dir = os.environ.get('MODEL_DIR', '/data/models')

print(f'Baixando {repo_id}...')
path = snapshot_download(
    repo_id=repo_id,
    local_dir=os.path.join(model_dir, repo_id.split('/')[-1]),
)
print(f'Modelo salvo em: {path}')
"

    if [ -d "${MODEL_PATH}" ]; then
        SIZE=$(du -sh "${MODEL_PATH}" | cut -f1)
        echo "[Download] Concluido! Tamanho: ${SIZE}"
    else
        echo "[Download] ERRO: Falha no download do modelo"
        echo "[Download] Verifique a conexao com a internet"
        exit 1
    fi
else
    SIZE=$(du -sh "${MODEL_PATH}" | cut -f1)
    echo "[Modelo] Encontrado em ${MODEL_PATH} (${SIZE})"
fi

echo "[Server] Iniciando MLC-LLM serve na porta ${PORT} com GPU..."
exec python3 -m mlc_llm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --device opencl
