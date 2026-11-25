#!/bin/bash
# Generic LoRA SFT continuation.
# usage: continue_sft_generic.sh <adapter> <data> <out> <steps> <gpu> [save_every]
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="$5"
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
mkdir -p "$3"
"$task_python" train/run_pixel_sft.py train \
  --model "$model" --adapter "$1" \
  --data "$2" --output "$3" \
  --steps "$4" --accum 2 --rank 16 --lr 2e-5 --lora-scope standard \
  --geometry-weight 1 --nonzero-weight 1 --gradient-checkpointing \
  --save-every "${6:-100}" --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
