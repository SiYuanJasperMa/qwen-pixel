#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
adapter_path=outputs/agent_sft_crop_feedback_paired_20261125/train/checkpoint-480
"$python_bin" -u scripts/eval_crop_recovery_continuations.py \
  --adapter "$adapter_path" --decode-policy fp32_state \
  --out outputs/agent_sft_crop_full_recovery_fp32_20261125
"$python_bin" -u scripts/eval_agent_tasks.py \
  --model "$model_path" --adapter "$adapter_path" \
  --tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --out outputs/agent_sft_crop_fp32_development_20261125 --split val \
  --max-steps 8 --max-new-tokens 4096 --system-profile clean_edit \
  --strict-extent --decode-policy fp32_state
