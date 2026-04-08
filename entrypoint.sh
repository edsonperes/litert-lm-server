#!/bin/bash
set -e

PORT="${PORT:-8000}"
MODEL="${MODEL:-RedPajama-INCITE-Chat-3B-v1-q4f16_1}"

echo "============================================"
echo "  MLC-LLM GPU Server para OrangePi 5"
echo "============================================"
echo "Modelo: ${MODEL}"
echo "Porta: ${PORT}"
echo "Device: Mali G610 (OpenCL)"
echo "============================================"

# Verificar GPU
echo "[GPU] Verificando OpenCL/Mali..."
if command -v clinfo &>/dev/null; then
    clinfo -l 2>/dev/null || echo "[GPU] AVISO: Nenhuma plataforma OpenCL detectada"
fi

echo "[Server] Iniciando API server na porta ${PORT}..."
cd /app
exec python3 server.py
