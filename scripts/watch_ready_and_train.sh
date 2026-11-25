#!/bin/bash
# Wait for <data>/READY and a free GPU, then launch training locally.
# usage: watch_ready_and_train.sh <data_dir> <out_dir> <steps> <gpu> <save_every>
set -u
DATA="$1"; OUT="$2"; STEPS="$3"; GPU="$4"; SAVE="${5:-200}"
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
LOG="$OUT/watch.log"
mkdir -p "$OUT"
while [ ! -f "$DATA/READY" ]; do sleep 120; done
echo "[$(date)] READY found" >> "$LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ] && break
  sleep 300
done
echo "[$(date)] launching on gpu$GPU" >> "$LOG"
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True QWEN_PIXEL_TRAIN_GPU="$GPU" \
  bash scripts/train_magicbrush_sft_27b.sh "$DATA/train.jsonl" "$OUT/train" "$STEPS" \
  /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B "$SAVE" \
  > "$OUT/train.log" 2>&1 < /dev/null &
