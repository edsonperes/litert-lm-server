# ============================================
# MLC-LLM GPU Server para OrangePi 5
# Base: milas/mlc-llm (Mali G610 OpenCL)
# ============================================
FROM docker.io/milas/mlc-llm:redpajama-3b

LABEL maintainer="edsonperes"
LABEL org.opencontainers.image.source="https://github.com/edsonperes/litert-lm-server"
LABEL org.opencontainers.image.description="MLC-LLM GPU inference server with OpenAI-compatible API for OrangePi 5"

# Instalar Python e deps para API server
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic

WORKDIR /app

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8000
ENV MODEL=RedPajama-INCITE-Chat-3B-v1-q4f16_1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/v1/models || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
