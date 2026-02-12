# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-trixie-slim

WORKDIR /workspace

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tar \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Combined steps to reduce image size:
# 1. Install pip packages
# 2. Clone PaddleSeg to /tmp, install it, extract configs, then remove source
# 3. Install rankseg
# 4. Clean uv cache
RUN uv pip install --system "numpy<2.0" \
    && uv pip install --system torch -i https://download.pytorch.org/whl/cpu \
    && uv pip install --system paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && git clone --depth 1 --single-branch --branch release/2.10 https://github.com/PaddlePaddle/PaddleSeg.git /tmp/paddleseg \
    && uv pip install --system /tmp/paddleseg \
    && cp -r /tmp/paddleseg/configs /workspace/configs \
    && rm -rf /tmp/paddleseg \
    && uv pip install --system git+https://github.com/Leev1s/rankseg.git@paddleseg \
    && uv cache clean

# Copy tools scripts (val.py, analyse.py, etc.)
COPY . /workspace

# Create 'download' command
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading CamVid dataset..."\n\
mkdir -p data\n\
curl -L https://paddleseg.bj.bcebos.com/dataset/camvid.tar -o data/camvid.tar\n\
(cd data && tar -xf camvid.tar && echo "Dataset Extracted")\n\
echo "Downloading pretrained models..."\n\
mkdir -p pretrained_models/pp_liteseg_camvid\n\
curl -L https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams -o pretrained_models/pp_liteseg_camvid/model-t.pdparams\n\
curl -L https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc2_camvid_960x720_10k/model.pdparams -o pretrained_models/pp_liteseg_camvid/model-b.pdparams\n\
' > /usr/local/bin/download && chmod +x /usr/local/bin/download

CMD ["/bin/bash"]
