#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/agent_teacher_scale_frozen_v1/train.jsonl --val data/agent_teacher_scale_frozen_v1/val.jsonl \
  --output outputs/agent_scale/sft --steps 400 --accum 2 --rank 16 --lr 7e-5 \
  --gradient-checkpointing --save-every 200 --eval-every 100 --eval-limit 4 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_scale/sft/checkpoint-400 \
  --tasks data/agent_teacher_scale_frozen_v1/tasks.jsonl --out outputs/agent_scale/eval_sft \
  --max-steps 8 --max-new-tokens 1200
