#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_output=outputs/rl_smoke_en_seed126_20261125
"$task_python" rl/run_oneshot_grpo.py \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter outputs/star_comparison_mixed_v1/train/checkpoint-1343 \
  --tasks data/rl_smoke_en_20261125/tasks_train.jsonl --out "$task_output/train" \
  --max-groups 16 --group-size 4 --updates 3 --max-new-tokens 6144 \
  --lr 1e-6 --beta 0.01 --seed 126 --temperature 0.7 --decode-policy deterministic
"$task_python" scripts/eval_local_qwen_oneshot.py \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter "$task_output/train/final_adapter" \
  --tasks data/rl_smoke_en_20261125/tasks_eval64.jsonl \
  --out "$task_output/eval" --limit 64 --max-new-tokens 6144 --decode-policy deterministic
