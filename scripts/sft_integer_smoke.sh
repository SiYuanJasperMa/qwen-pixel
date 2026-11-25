#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=5
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
"$task_python" scripts/audit_clean_training_rows.py --data data/sft_integer_smoke_v1/train.jsonl
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/sft_integer_smoke_v1/train.jsonl --output outputs/sft_integer_smoke_v1/train \
  --steps 120 --accum 1 --rank 16 --lr 2e-4 --lora-scope standard --geometry-weight 3 \
  --nonzero-weight 1 --gradient-checkpointing --save-every 120 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144
"$task_python" scripts/check_training_prefixes.py --model "$task_model" \
  --adapter outputs/sft_integer_smoke_v1/train/checkpoint-120 --data data/sft_integer_smoke_v1/train.jsonl \
  --out outputs/sft_integer_smoke_v1/prefixes.json --limit 0 --max-new-tokens 1800
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/sft_integer_smoke_v1/train/checkpoint-120 --tasks data/sft_integer_smoke_v1/tasks.jsonl \
  --split train --out outputs/sft_integer_smoke_v1/rollout --max-steps 8 \
  --max-new-tokens 1800 --system-profile integer_edit --strict-extent
