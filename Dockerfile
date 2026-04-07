# ============================================
# Stage 1: Builder - instala dependencias Python
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
# Stage 2: Runtime - imagem final leve
# ============================================
FROM debian:bookworm-slim

LABEL maintainer="edsonperes"
LABEL org.opencontainers.image.source="https://github.com/edsonperes/litert-lm-server"
LABEL org.opencontainers.image.description="LiteRT-LM inference server with OpenAI-compatible API for OrangePi 5"

# Deps runtime: Python, OpenCL, libs graficas para Mali
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    ocl-icd-libopencl1 \
    clinfo \
    libwayland-client0 \
    libwayland-server0 \
    libx11-xcb1 \
    libxcb-dri2-0 \
    libxcb-dri3-0 \
    libdrm2 \
    libgbm1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download libmali para Mali G610 (RK3588/OrangePi 5)
RUN mkdir -p /etc/OpenCL/vendors && \
    wget -q -O /usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so \
    "https://github.com/JeffyCN/mirrors/raw/libmali/lib/aarch64-linux-gnu/libmali-valhall-g610-g6p0-x11-wayland-gbm.so" && \
    echo "/usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so" > /etc/OpenCL/vendors/mali.icd && \
    ln -sf /usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so /usr/lib/libmali.so && \
    ln -sf /usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so /usr/lib/libOpenCL.so.1 && \
    ldconfig

# Copiar venv do builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Diretorio de trabalho
WORKDIR /app

# Copiar codigo da aplicacao
COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Variaveis de ambiente padrao
ENV MODEL_DIR=/data/models
ENV MODEL_FILE=gemma-4-E2B-it.litertlm
ENV MODEL_REPO=litert-community/gemma-4-E2B-it-litert-lm
ENV MODEL_ID=gemma-4-E2B-it
ENV PORT=8000
ENV INFERENCE_BACKEND=python
ENV BACKEND_TYPE=cpu
ENV HUGGING_FACE_HUB_TOKEN=""

# Volume para persistencia do modelo
VOLUME /data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
