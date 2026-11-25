#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="${QWEN_PIXEL_TRAIN_GPU:-0}"
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_variant="${1:-sft_clean_mask_v3}"
test -f "data/$task_variant/train_verified.jsonl"
test -f "data/$task_variant/reference_review_manifest.json"
test -f "data/$task_variant/development_tasks.jsonl"
"$task_python" train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B \
  --data "data/$task_variant/train_verified.jsonl" --output "outputs/$task_variant/train" \
  --steps 1200 --accum 2 --rank 16 --lr 5e-5 --lora-scope standard \
  --geometry-weight 3 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 300 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144 \
  --checkpoint-hook scripts/clean_checkpoint_eval.py
