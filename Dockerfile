# Base Image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Fix DNS resolution inside Fly.io builder
RUN echo "nameserver 1.1.1.1" > /etc/resolv.conf

# Metadata
LABEL maintainer="sumesh@meledath.me"

# Core Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    MODE=serve \
    PATH="/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CFLAGS="-mcmodel=large" \
    CXXFLAGS="-mcmodel=large" \
    LDFLAGS="-mcmodel=large" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# System + Python + GCC 13 + CUDA libs
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y \
        gcc-13 g++-13 \
        build-essential cmake \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git wget curl sqlite3 libopenblas-dev ca-certificates \
        cuda-cudart-dev-12-8 \
        cuda-nvrtc-dev-12-8 \
        cuda-driver-dev-12-8 \
        file && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Install Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        typing_extensions \
        pyyaml \
        langchain \
        langchain-community \
        faiss-cpu \
        sentence-transformers \
        tiktoken

# Torch from source (optional, pinned to RTX 5080 arch)
ENV USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_MKLDNN=1 \
    USE_MKL=0 \
    MAX_JOBS=16 \
    TORCH_CUDA_ARCH_LIST="8.9+PTX;9.0;12.0"

# Download Torch + cuDNN from Cloudflare Tunnel
RUN wget -O /tmp/torch.whl http://files-public.desknav.ai/llm/torch.whl && \
    MIME_TYPE=$(file -b --mime-type /tmp/torch.whl) && \
    echo "Downloaded torch.whl - MIME type: $MIME_TYPE" && \
    test "$MIME_TYPE" = "application/zip" && \
    mv /tmp/torch.whl /tmp/torch-2.8.0a0-cp310-cp310-linux_x86_64.whl && \
    pip install --no-deps /tmp/torch-2.8.0a0-cp310-cp310-linux_x86_64.whl

RUN wget -q -O /tmp/cudnn.tar.xz http://files-public.desknav.ai/llm/cudnn.tar.xz && \
    tar -xf /tmp/cudnn.tar.xz -C /tmp && \
    CUDNN_DIR=$(find /tmp -type d -name "cudnn-linux-x86_64*" | head -n 1) && \
    cp -P "$CUDNN_DIR/include/"* /usr/include/ && \
    cp -P "$CUDNN_DIR/lib/"* /usr/lib/x86_64-linux-gnu/ && \
    echo "/usr/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/cudnn.conf && \
    ldconfig && \
    rm -rf /tmp/cudnn*

# Build bitsandbytes from source
WORKDIR /opt
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git bnb
WORKDIR /opt/bnb
RUN cmake -DCOMPUTE_BACKEND=cuda -S . && \
    make -j$(nproc) && \
    pip install -e .

# Copy application
COPY . /app
WORKDIR /app

# App requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pymupdf python-docx sentencepiece python-multipart
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install gradio
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "huggingface_hub[hf_xet]"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install psutil

# Final setup
RUN chmod +x /app/start.sh
CMD ["bash", "./start.sh"]
