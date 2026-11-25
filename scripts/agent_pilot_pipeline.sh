#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=${PILOT_GPU:-2}
export PYTHONUNBUFFERED=1
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/agent_teacher_pilot_v1/train.jsonl --val data/agent_teacher_pilot_v1/val.jsonl \
  --output outputs/agent_pilot/sft --steps 100 --accum 1 --rank 16 --lr 1e-4 \
  --gradient-checkpointing --save-every 100 --eval-every 50 --eval-limit 3 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_pilot/sft/checkpoint-100 \
  --tasks data/agent_teacher_pilot_v1/tasks.jsonl --out outputs/agent_pilot/eval_sft --limit 3
"$task_python" rl/run_agent_grpo.py --model "$task_model" \
  --adapter outputs/agent_pilot/sft/checkpoint-100 --tasks data/agent_teacher_pilot_v1/tasks.jsonl \
  --out outputs/agent_pilot/grpo --groups 3 --group-size 3 --max-steps 6 --max-new-tokens 1000
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_pilot/grpo/final_adapter \
  --tasks data/agent_teacher_pilot_v1/tasks.jsonl --out outputs/agent_pilot/eval_rl --limit 3
