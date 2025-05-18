#!/bin/bash

echo "🛑 Stopping lora-combined systemd service..."
sudo systemctl stop lora-combined.service

echo "🐳 Checking and stopping any running Docker container..."
docker ps | grep lora-combined && docker stop lora-combined

echo "🚀 Starting systemd service again..."
sudo systemctl start lora-combined.service

echo "📡 Waiting 3 seconds for startup..."
sleep 3

echo "🔍 Checking service status..."
sudo systemctl status lora-combined.service

echo "🌐 Verifying FastAPI is live..."
curl http://localhost:8000
