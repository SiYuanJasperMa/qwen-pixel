#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/sft_seven_tasks_v1/train.jsonl --output outputs/sft_seven_tasks_v1/train \
  --steps 800 --accum 1 --rank 16 --lr 1e-4 --lora-scope standard \
  --gradient-checkpointing --save-every 400 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
"$task_python" scripts/check_training_prefixes.py --model "$task_model" \
  --adapter outputs/sft_seven_tasks_v1/train/checkpoint-800 \
  --data data/sft_seven_tasks_v1/train.jsonl --out outputs/sft_seven_tasks_v1/prefixes.json \
  --limit 0 --max-new-tokens 1600
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/sft_seven_tasks_v1/train/checkpoint-800 \
  --tasks data/sft_seven_tasks_v1/tasks.jsonl --split val --out outputs/sft_seven_tasks_v1/val_rollouts \
  --max-steps 8 --system-profile rgb
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/sft_seven_tasks_v1/train/checkpoint-800 \
  --tasks data/sft_seven_tasks_v1/tasks.jsonl --split train --out outputs/sft_seven_tasks_v1/train_rollouts \
  --max-steps 8 --system-profile rgb
