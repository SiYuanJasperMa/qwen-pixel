#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
condition=${1:?Specify crop_recovery or ordinary_control}
case "$condition" in crop_recovery|ordinary_control) ;; *) exit 2;; esac
export CUDA_VISIBLE_DEVICES=${2:?Specify verified free physical GPU}
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
output_dir=outputs/agent_sft_mask_crop_pair_20261125/$condition
if [[ -e "$output_dir/train" ]]; then
  echo 'Inspect recorded run before restarting; preserve existing training.' >&2
  exit 2
fi
"$python_bin" -u train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter outputs/agent_sft_en_27b_37photos_geometry4_exposure30_20261125/train/checkpoint-1600 \
  --data "data/agent_sft_mask_and_crop_comparison_20261125/$condition/train.jsonl" \
  --output "$output_dir/train" --steps 480 --accum 2 --rank 16 --lr 2e-5 \
  --seed 42 --lora-scope attention --geometry-weight 1 --nonzero-weight 1 \
  --gradient-checkpointing --save-every 240 --eval-every 0 \
  --keep-checkpoint-training-state --max-seq-len 8192 --image-max-pixels 262144 \
  --agent-val-tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --agent-val-profile clean_edit --agent-val-every 240 --agent-val-decode-policy deterministic
