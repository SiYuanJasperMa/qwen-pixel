#!/bin/bash
# LoRA SFT with rank 32 (capacity ablation).
# usage: train_sft_r32.sh <data> <out> <steps> <gpu>
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="$4"
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
mkdir -p "$2"
"$task_python" train/run_pixel_sft.py train \
  --model "$model" \
  --data "$1" --output "$2" \
  --steps "$3" --accum 2 --rank 32 --lr 2e-5 --lora-scope standard \
  --geometry-weight 1 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 200 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
