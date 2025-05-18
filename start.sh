#!/bin/bash

cd /app

if [ "$MODE" = "train" ]; then
    echo "🎯 MODE=train: Starting LoRA training..."
    python3 train_lora_v2.py
else
    echo "🚀 MODE=serve: Starting FastAPI LLM Server + Telegram bot..."
    python3 start_combined.py
fi
