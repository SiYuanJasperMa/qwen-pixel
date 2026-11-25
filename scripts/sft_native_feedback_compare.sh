#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
condition=${1:?Specify feedback or control}
case "$condition" in feedback|control) ;; *) exit 2;; esac
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
data_dir=data/agent_sft_native_feedback_comparison_20261125/$condition
output_dir=${2:-outputs/agent_sft_native_feedback_${condition}_20261125}
if [[ -e "$output_dir/train" ]]; then
  echo 'Inspect the recorded process; do not overwrite an existing training run.' >&2
  exit 2
fi
"$python_bin" -u scripts/audit_sft_dataset_budget.py --data "$data_dir/train.jsonl"
"$python_bin" -u train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter outputs/agent_sft_en_27b_37photos_geometry4_exposure30_20261125/train/checkpoint-1600 \
  --data "$data_dir/train.jsonl" --output "$output_dir/train" \
  --steps 4500 --accum 2 --rank 16 --lr 2e-5 --seed 42 --lora-scope attention \
  --geometry-weight 1 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 450 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144 \
  --agent-val-tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --agent-val-profile clean_edit --agent-val-every 450 --agent-val-decode-policy deterministic
