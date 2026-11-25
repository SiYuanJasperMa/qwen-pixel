#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
for condition in geometry1_control geometry4; do
  "$python_bin" -u scripts/audit_visual_conditioning.py \
    --adapter "outputs/agent_sft_en_27b_37photos_${condition}_20261125/train/checkpoint-400" \
    --data data/agent_sft_en_27b_37photos_20261125/train.jsonl --images 6 \
    --out "outputs/agent_sft_en_27b_37photos_${condition}_20261125/visual_conditioning"
done
