#!/bin/bash
# Wait for v4 checkpoint-1600 and a free GPU 1, then continue v4 for +800 steps.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 99999)
  if [ -d outputs/sft_magicbrush_27b_v4/train/checkpoint-1600 ] && [ "$used" -lt 5000 ]; then
    break
  fi
  sleep 60
done
mkdir -p outputs/sft_magicbrush_27b_v4_cont
setsid nohup env QWEN_PIXEL_TRAIN_GPU=1 bash scripts/continue_27b_v4.sh 800 \
  > outputs/sft_magicbrush_27b_v4_cont/train.log 2>&1 < /dev/null &
echo "v4 continuation launched"
