# ============================================
# LiteRT-LM inference server for OrangePi 5
# Gemma 4 E2B - CPU optimized
# ============================================
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ============================================
# Runtime
# ============================================
FROM debian:bookworm-slim

LABEL maintainer="edsonperes"
LABEL org.opencontainers.image.source="https://github.com/edsonperes/litert-lm-server"
LABEL org.opencontainers.image.description="LiteRT-LM Gemma 4 E2B inference server for OrangePi 5"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /app

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_DIR=/data/models
ENV MODEL_FILE=gemma-4-E2B-it.litertlm
ENV MODEL_REPO=litert-community/gemma-4-E2B-it-litert-lm
ENV MODEL_ID=gemma-4-E2B-it
ENV PORT=8000

VOLUME /data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/v1/models || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
