#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=3
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
"$task_python" train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B \
  --adapter outputs/sft_clean_mask_v4_raw/train/checkpoint-1200 \
  --data data/sft_clean_mask_v4_raw/train_verified.jsonl --output outputs/sft_extended_v1/train \
  --steps 4000 --accum 2 --rank 16 --lr 3e-5 --warmup 25 --lora-scope standard \
  --geometry-weight 3 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 400 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144 \
  --checkpoint-hook scripts/clean_checkpoint_eval.py
