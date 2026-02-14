##### Stage 1: Builder
# Use uv image for fast dependency installation
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /build

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install Python dependencies into virtual environment /build/.venv
# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
# Step-by-step installation to leverage caching
# Install CPU versions of torch and paddle to save significant space
RUN uv pip install --no-cache numpy==1.26.4 \
    && uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --no-cache paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# ------------------------------------------------------------------------------------------------

##### Stage 2: Pioneer (frequently changing layer)
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS pioneer

WORKDIR /tmp

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Clone repository (depth 1 for speed and minimal size)
RUN git clone --depth 1 --single-branch --branch release/2.10 https://github.com/PaddlePaddle/PaddleSeg.git /tmp/paddleseg

# Clone rankseg repository
COPY . /tmp/rankseg

# Install git dependencies
RUN uv pip install --no-cache /tmp/paddleseg \
    && uv pip install --no-cache /tmp/rankseg

# ------------------------------------------------------------------------------------------------

##### Stage 3: Runtime
# Use standard python slim image for minimal runtime size (no build tools)
FROM python:3.10-slim-bookworm

WORKDIR /workspace

# Install necessary runtime system libraries 
# OpenCV requires libgl1, libglib2.0-0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from pioneer stage
COPY --from=pioneer /opt/venv /opt/venv

# Set environment variables to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy configuration files from pioneer stage
COPY --from=pioneer /tmp/paddleseg/configs /workspace/configs

# Copy tools scripts (val.py, analyse.py, etc.)
COPY rankseg/paddleseg/tools /workspace

# Create directories for data and pretrained models
RUN mkdir -p /workspace/data && mkdir -p /workspace/pretrained_models

CMD ["/bin/bash"]
