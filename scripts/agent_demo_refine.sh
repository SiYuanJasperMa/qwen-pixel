#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_model=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
"$task_python" train/run_pixel_sft.py train --model "$task_model" \
  --adapter outputs/agent_demo/sft/checkpoint-30 \
  --data data/agent_demo_v1/train.jsonl --output outputs/agent_demo/sft_refine \
  --steps 120 --accum 1 --rank 16 --lr 2e-4 --gradient-checkpointing \
  --save-every 120 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1
"$task_python" scripts/run_agent_demo.py --model "$task_model" \
  --adapter outputs/agent_demo/sft_refine/checkpoint-120 \
  --source outputs/pixel_edit_demo/source.png \
  --instruction 'Change the red bodywork of both buses to blue. Preserve the windows, wheels, road, and background.' \
  --out outputs/agent_demo/qwen_refine --max-steps 10 --max-new-tokens 1800
