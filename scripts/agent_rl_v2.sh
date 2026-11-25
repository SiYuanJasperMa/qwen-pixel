#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
"$task_python" rl/run_agent_grpo.py --model "$task_model" \
  --adapter outputs/agent_pilot/sft/checkpoint-100 --tasks data/agent_teacher_pilot_v1/tasks.jsonl \
  --out outputs/agent_pilot_v2/grpo --groups 2 --group-size 3 --max-steps 8 --max-new-tokens 1000
"$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_pilot_v2/grpo/final_adapter \
  --tasks data/agent_teacher_pilot_v1/tasks.jsonl --out outputs/agent_pilot_v2/eval_rl --limit 3 --max-steps 8
