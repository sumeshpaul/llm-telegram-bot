#!/bin/bash

# === CONFIG ===
SOURCE_DIR="/home/sam/projects/lora-finetune"
DEST_DIR="/mnt/nas_llm_backup/backups"
DATE_TAG=$(date +"%Y%m%d_%H%M")
ARCHIVE_NAME="lora-finetune_backup_$DATE_TAG.tar.gz"
LOG_FILE="/home/sam/lora_backup_archive.log"

# === ENSURE DESTINATION FOLDER EXISTS ===
mkdir -p "$DEST_DIR"

# === START LOG ===
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting backup: $ARCHIVE_NAME" >> "$LOG_FILE"

# === CREATE TAR.GZ ARCHIVE ===
tar -czf "$DEST_DIR/$ARCHIVE_NAME" -C "$SOURCE_DIR" . >> "$LOG_FILE" 2>&1

# === FINISH LOG ===
if [ $? -eq 0 ]; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Backup successful: $ARCHIVE_NAME" >> "$LOG_FILE"
else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] ❌ Backup FAILED: $ARCHIVE_NAME" >> "$LOG_FILE"
fi

echo "------------------------------------------------" >> "$LOG_FILE"
