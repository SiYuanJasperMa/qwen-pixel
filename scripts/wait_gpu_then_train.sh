#!/bin/bash
# Wait until the given physical GPU has no large allocation, then start 27B LoRA SFT.
# usage: wait_gpu_then_train.sh <gpu> <data> <out> <steps>
set -u
gpu="$1"; data="$2"; out="$3"; steps="$4"
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
while [ ! -f "$data" ]; do sleep 60; done
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ] && break
  sleep 60
done
mkdir -p "$out"
QWEN_PIXEL_TRAIN_GPU="$gpu" nohup bash scripts/train_magicbrush_sft_27b.sh \
  "$data" "$out" "$steps" > "$out/train.log" 2>&1 &
echo "launched gpu=$gpu data=$data out=$out steps=$steps"
