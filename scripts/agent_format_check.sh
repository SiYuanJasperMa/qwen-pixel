#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/agent_rgb_teacher_user_v11/train.jsonl --val data/agent_rgb_teacher_user_v11/val.jsonl \
  --output outputs/agent_format_v11/sft --steps 50 --accum 2 --rank 16 --lr 1e-5 \
  --lora-scope standard --gradient-checkpointing --save-every 50 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_format_v11/sft/checkpoint-50 \
  --tasks data/agent_native_v4/tasks.jsonl --out outputs/agent_format_v11/eval \
  --limit 3 --max-steps 8 --system-profile rgb --tool-image-mode user
