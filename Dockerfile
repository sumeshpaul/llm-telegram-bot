# Base Image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Metadata
LABEL maintainer="sumesh@meledath.me"

# Core Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    MODE=serve \
    PATH="/venv/bin:$PATH" \
    CFLAGS="-mcmodel=large" \
    CXXFLAGS="-mcmodel=large" \
    LDFLAGS="-mcmodel=large" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# System + Python + GCC 13
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y \
        gcc-13 g++-13 \
        build-essential cmake \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git wget curl sqlite3 libopenblas-dev ca-certificates file && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python packages
RUN pip install \
        typing_extensions \
        pyyaml \
        langchain \
        langchain-community \
        faiss-cpu \
        sentence-transformers \
        tiktoken

# ✅ Download 5080-compatible PyTorch wheel from public URL
RUN wget -O /tmp/torch.whl https://files-public.desknav.ai/llm/torch-latest.whl && \
    pip install --no-deps /tmp/torch.whl

RUN wget -O /tmp/cudnn.tar.xz http://files-public.desknav.ai/llm/cudnn.tar.xz && \
    tar -xf /tmp/cudnn.tar.xz -C /tmp && \
    CUDNN_DIR=$(find /tmp -type d -name "cudnn-linux-x86_64*" | head -n 1) && \
    cp -P "$CUDNN_DIR/include/"* /usr/include/ && \
    cp -P "$CUDNN_DIR/lib/"* /usr/lib/x86_64-linux-gnu/ && \
    echo "/usr/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/cudnn.conf && \
    ldconfig && \
    rm -rf /tmp/cudnn*

# bitsandbytes build
WORKDIR /opt
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git bnb
WORKDIR /opt/bnb
RUN cmake -DCOMPUTE_BACKEND=cuda -S . && \
    make -j$(nproc) && \
    pip install -e .

# Copy app
COPY . /app
WORKDIR /app

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pymupdf python-docx sentencepiece python-multipart
RUN pip install gradio

# Finalize
RUN chmod +x /app/start.sh
CMD ["bash", "./start.sh"]
