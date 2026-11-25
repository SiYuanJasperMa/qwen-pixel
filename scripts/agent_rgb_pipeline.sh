#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --data data/agent_rgb_teacher_v6/train.jsonl --val data/agent_rgb_teacher_v6/val.jsonl \
  --output outputs/agent_rgb_v6/sft_rgb --steps 300 --accum 2 --rank 16 --lr 7e-5 \
  --gradient-checkpointing --save-every 300 --eval-every 100 --eval-limit 4 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
CUDA_VISIBLE_DEVICES=5 "$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_rgb_v6/sft_rgb/checkpoint-300 \
  --tasks data/agent_native_v4/tasks.jsonl --out outputs/agent_rgb_v6/eval_sft_rgb \
  --max-steps 8 --max-new-tokens 1200 --system-profile rgb > outputs/agent_rgb_v6/eval_sft_rgb.log 2>&1 &
task_rgb_sft_eval_pid=$!
CUDA_VISIBLE_DEVICES=6 "$task_python" rl/run_agent_grpo.py --model "$task_model" \
  --adapter outputs/agent_rgb_v6/sft_rgb/checkpoint-300 --tasks data/agent_native_v4/tasks.jsonl \
  --out outputs/agent_rgb_v6/grpo_rgb --groups 2 --group-size 3 --max-steps 8 --max-new-tokens 1000 --system-profile rgb
CUDA_VISIBLE_DEVICES=6 "$task_python" scripts/eval_agent_tasks.py --model "$task_model" \
  --adapter outputs/agent_rgb_v6/grpo_rgb/final_adapter \
  --tasks data/agent_native_v4/tasks.jsonl --out outputs/agent_rgb_v6/eval_rl_rgb \
  --max-steps 8 --max-new-tokens 1200 --system-profile rgb
wait "$task_rgb_sft_eval_pid"
