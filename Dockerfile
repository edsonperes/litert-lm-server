# ============================================
# MLC-LLM inference server for OrangePi 5
# Mali G610 GPU acceleration via OpenCL
# ============================================
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
    --pre mlc-llm \
    -f https://mlc.ai/wheels && \
    /opt/venv/bin/pip install --no-cache-dir huggingface-hub

# ============================================
# Runtime
# ============================================
FROM debian:bookworm-slim

LABEL maintainer="edsonperes"
LABEL org.opencontainers.image.source="https://github.com/edsonperes/litert-lm-server"
LABEL org.opencontainers.image.description="MLC-LLM inference server with GPU acceleration for OrangePi 5"

# Runtime deps: Python, OpenCL, Mali GPU libs
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
    libvulkan1 \
    mesa-vulkan-drivers \
    wget \
    curl \
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

WORKDIR /app

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Variaveis de ambiente
ENV MODEL_REPO=mlc-ai/gemma-2b-it-q4f16_1-MLC
ENV MODEL_DIR=/data/models
ENV PORT=8000

VOLUME /data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/v1/models || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
