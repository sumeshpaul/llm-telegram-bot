FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# --- ENV Configuration ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TOKENIZERS_PARALLELISM=false
ENV MODE=serve
ENV PATH="/venv/bin:$PATH"
ENV CFLAGS="-mcmodel=large"
ENV CXXFLAGS="-mcmodel=large"
ENV LDFLAGS="-mcmodel=large"

# --- System + Python Setup ---
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl sqlite3 libopenblas-dev ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --upgrade pip && \
    pip install typing_extensions pyyaml

# --- Install cuDNN from Local Archive ---
COPY cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz /tmp/
RUN tar -xf /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz -C /tmp && \
    cp -P /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/include/ && \
    cp -P /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/lib/x86_64-linux-gnu/ && \
    echo "/usr/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/cudnn.conf && \
    ldconfig && \
    rm -rf /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive*

# --- Create Virtual Environment ---
RUN python -m venv /venv

# --- Install Custom PyTorch Wheel ---
COPY torch-2.8.0a0-cp310-cp310-linux_x86_64.whl /tmp/
RUN /venv/bin/pip install /tmp/torch-2.8.0a0-cp310-cp310-linux_x86_64.whl --no-deps

# --- Install Python Dependencies ---
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /venv/bin/pip install --no-cache-dir tiktoken protobuf blobfile

# --- Build bitsandbytes from source ---
RUN git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && /venv/bin/pip install . && cd .. && rm -rf bitsandbytes

# --- App Setup ---
WORKDIR /app
COPY . .
RUN chmod +x /app/start.sh /app/train_lora_v2.py && mkdir -p /app/data /app/backups

# --- Networking + Entrypoint ---
EXPOSE 8000
CMD if [ "$MODE" = "train" ]; then \
      python3 /app/train_lora_v2.py; \
    else \
      bash /app/start.sh; \
    fi
