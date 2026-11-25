#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="${QWEN_PIXEL_TRAIN_GPU:-1}"
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
data="${1:-data/sft_magicbrush_gpt61_inpaint_54_v1/train.jsonl}"
out="${2:-outputs/sft_magicbrush_inpaint_v1/train}"
steps="${3:-300}"
mkdir -p "$out"
"$task_python" train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B \
  --data "$data" --output "$out" \
  --steps "$steps" --accum 2 --rank 16 --lr 5e-5 --lora-scope standard \
  --geometry-weight 1 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 100 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
