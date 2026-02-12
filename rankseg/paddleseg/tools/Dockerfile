# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-trixie-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    tar \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN uv pip install --system paddlepaddle==3.3.0

# Install rankseg from git since we are building from tools context
RUN uv pip install --system git+https://github.com/Leev1s/rankseg.git@paddleseg

# Copy tools scripts (val.py, analyse.py, etc.)
COPY . /app

# Create 'download' command (keeps the complex curl/tar logic encapsulated)
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading CamVid dataset..."\n\
mkdir -p data\n\
curl -L https://paddleseg.bj.bcebos.com/dataset/camvid.tar -o data/camvid.tar\n\
(cd data && tar -xf camvid.tar && echo "Dataset Extracted")\n\
\n\
echo "Downloading pretrained models..."\n\
mkdir -p pretrained_models/pp_liteseg_camvid\n\
curl -L https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams -o pretrained_models/pp_liteseg_camvid/model-t.pdparams\n\
curl -L https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc2_camvid_960x720_10k/model.pdparams -o pretrained_models/pp_liteseg_camvid/model-b.pdparams\n\
' > /usr/local/bin/download && chmod +x /usr/local/bin/download

CMD ["/bin/bash"]
