#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/data/models}"
MODEL_FILE="${MODEL_FILE:-gemma-4-E2B-it.litertlm}"
MODEL_REPO="${MODEL_REPO:-litert-community/gemma-4-E2B-it-litert-lm}"
PORT="${PORT:-8000}"
BACKEND_TYPE="${BACKEND_TYPE:-cpu}"

echo "============================================"
echo "  LiteRT-LM Server para OrangePi 5"
echo "============================================"
echo "Modelo: ${MODEL_REPO}/${MODEL_FILE}"
echo "Backend: ${INFERENCE_BACKEND:-python} (${BACKEND_TYPE})"
echo "Porta: ${PORT}"
echo "============================================"

mkdir -p "${MODEL_DIR}"

# Verificar GPU se backend=gpu
if [ "${BACKEND_TYPE}" = "gpu" ]; then
    echo "[GPU] Verificando OpenCL/Mali..."
    if command -v clinfo &>/dev/null; then
        PLATFORMS=$(clinfo -l 2>/dev/null | grep -c "Platform" || true)
        if [ "${PLATFORMS}" -gt 0 ]; then
            echo "[GPU] OpenCL disponivel - ${PLATFORMS} plataforma(s) detectada(s)"
            clinfo -l 2>/dev/null || true
        else
            echo "[GPU] AVISO: Nenhuma plataforma OpenCL detectada"
            echo "[GPU] Verifique se /lib/firmware/mali_csffw.bin esta montado"
            echo "[GPU] Continuando sem aceleracao GPU..."
        fi
    else
        echo "[GPU] AVISO: clinfo nao encontrado, impossivel verificar GPU"
    fi
fi

# Download do modelo se nao existir
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
if [ ! -f "${MODEL_PATH}" ]; then
    echo "[Download] Modelo nao encontrado em ${MODEL_PATH}"
    echo "[Download] Baixando de ${MODEL_REPO}..."
    echo "[Download] Isso pode levar alguns minutos (~2.6GB)..."

    python3 -c "
from huggingface_hub import hf_hub_download
import os

repo_id = os.environ.get('MODEL_REPO', 'litert-community/gemma-4-E2B-it-litert-lm')
filename = os.environ.get('MODEL_FILE', 'gemma-4-E2B-it.litertlm')
model_dir = os.environ.get('MODEL_DIR', '/data/models')

print(f'Baixando {filename} de {repo_id}...')
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)
print(f'Modelo salvo em: {path}')
"

    if [ -f "${MODEL_PATH}" ]; then
        SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
        echo "[Download] Concluido! Tamanho: ${SIZE}"
    else
        echo "[Download] ERRO: Falha no download do modelo"
        echo "[Download] Verifique a conexao com a internet"
        exit 1
    fi
else
    SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
    echo "[Modelo] Encontrado em ${MODEL_PATH} (${SIZE})"
fi

echo "[Server] Iniciando servidor na porta ${PORT}..."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --log-level info
