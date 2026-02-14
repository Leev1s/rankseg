##### Stage 1: Downloader
# Use a lightweight git image to clone repositories
FROM alpine/git AS downloader

WORKDIR /tmp
# Clone repository (depth 1 for speed and minimal size)
RUN git clone --depth 1 --single-branch --branch release/2.10 https://github.com/PaddlePaddle/PaddleSeg.git /tmp/paddleseg

# Clone rankseg repository
RUN git clone --depth 1 --single-branch --branch paddleseg https://github.com/Leev1s/rankseg.git /tmp/rankseg

# ------------------------------------------------------------------------------------------------

##### Stage 2: Builder
# Use uv image for fast dependency installation
FROM ghcr.io/astral-sh/uv:python3.10-trixie-slim AS builder

WORKDIR /build

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install Python dependencies into virtual environment /build/.venv

# 1. Create virtual environment
RUN uv venv /build/.venv

# 2. Copy repositories
COPY --from=downloader /tmp/paddleseg /build/paddleseg
COPY --from=downloader /tmp/rankseg /build/rankseg

# 3. Activate virtual environment
ENV PATH="/build/.venv/bin:$PATH"

# 4. Install dependencies
# Step-by-step installation to leverage caching
# Install CPU versions of torch and paddle to save significant space
# Install heavy dependencies first (cached layer)
RUN uv pip install --no-cache "numpy<2.0" \
    && uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --no-cache paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install git dependencies (frequently changing layer)
RUN uv pip install --no-cache /build/paddleseg \
    && uv pip install --no-cache /build/rankseg

# ------------------------------------------------------------------------------------------------

##### Stage 3: Runtime
# Use standard python slim image for minimal runtime size (no build tools)
FROM python:3.10-slim-trixie

WORKDIR /workspace

# Install necessary runtime system libraries 
# OpenCV requires libgl1, libglib2.0-0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /build/.venv /workspace/.venv

# Copy configuration files from downloader stage
COPY --from=downloader /tmp/paddleseg/configs /workspace/configs

# Set environment variables to use virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"

# Copy tools scripts (val.py, analyse.py, etc.)
COPY . /workspace

# Create directories for data and pretrained models
RUN mkdir -p /workspace/data && mkdir -p /workspace/pretrained_models

CMD ["/bin/bash"]
