#!/bin/bash
# Method comparison: same initialization/serializer/assistant-token budget.
# usage: run_star_comparison.sh rft|mixed gpu
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="$2"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$1" in rft|mixed) ;; *) exit 2;; esac
task_seed="${3:-42}"
task_suffix=v1
if [ "$task_seed" != 42 ]; then task_suffix="seed${task_seed}"; fi
exec /mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python \
  train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --data "data/star_clean_v2/sft_${1}_aug/train.jsonl" \
  --output "outputs/star_comparison_${1}_${task_suffix}/train" \
  --steps 6000 --supervised-token-budget 8000000 --accum 2 \
  --rank 16 --lora-scope standard --lr 2e-5 --seed "$task_seed" \
  --nonzero-weight 1 --geometry-weight 1 --gradient-checkpointing \
  --save-every 500 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
