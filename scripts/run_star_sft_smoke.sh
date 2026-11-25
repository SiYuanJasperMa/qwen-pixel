#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${QWEN_PIXEL_SFT_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/star_clean_v1/sft_rft/train.jsonl --limit 16 \
  --output outputs/star_sft_contract_smoke_v1/train \
  --steps 10 --accum 2 --rank 16 --lr 2e-5 --lora-scope standard \
  --nonzero-weight 1 --geometry-weight 1 --gradient-checkpointing \
  --save-every 10 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144
"$task_python" scripts/eval_local_qwen_oneshot.py --model "$task_model" \
  --adapter outputs/star_sft_contract_smoke_v1/train/checkpoint-10 \
  --tasks data/star_clean_v1/smoke_eval_tasks.jsonl \
  --out outputs/star_sft_contract_smoke_v1/eval --limit 2 --max-new-tokens 4096
