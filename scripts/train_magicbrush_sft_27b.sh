#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="${QWEN_PIXEL_TRAIN_GPU:-1}"
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model="${4:-/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B}"
data="${1:-data/sft_magicbrush_scale_v1/train.jsonl}"
out="${2:-outputs/sft_magicbrush_27b_v1/train}"
steps="${3:-200}"
save_every="${5:-50}"
mkdir -p "$out"
"$task_python" train/run_pixel_sft.py train \
  --model "$model" \
  --data "$data" --output "$out" \
  --steps "$steps" --accum 2 --rank 16 --lr 2e-5 --lora-scope standard \
  --geometry-weight 1 --nonzero-weight 1 --gradient-checkpointing \
  --save-every "$save_every" --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
