#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
"$python_bin" -u scripts/audit_crop_prefix_restoration.py \
  --out outputs/agent_sft_crop_prefix_restoration_20261125
"$python_bin" -u scripts/eval_crop_recovery_continuations.py \
  --adapter outputs/agent_sft_crop_feedback_paired_20261125/train/checkpoint-480 \
  --out outputs/agent_sft_crop_full_recovery_paired_20261125
"$python_bin" -u scripts/eval_crop_recovery_continuations.py \
  --adapter outputs/agent_sft_crop_feedback_positive_control_20261125/train/checkpoint-480 \
  --out outputs/agent_sft_crop_full_recovery_positive_20261125
