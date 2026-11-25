#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${QWEN_PIXEL_RL_GPU:-0}"
exec /mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python \
  rl/run_oneshot_grpo.py \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter outputs/sft_all_v14_cont/train/checkpoint-2300 \
  --tasks data/star_rl_smoke_v1/tasks.jsonl \
  --out outputs/star_rl_smoke_v2 \
  --max-groups 8 --group-size 4 --updates 1 --max-new-tokens 4096 \
  --lr 1e-6 --beta 0.01 --seed 123
