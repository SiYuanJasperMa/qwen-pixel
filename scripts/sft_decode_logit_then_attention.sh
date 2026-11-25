#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
adapter=outputs/agent_sft_en_27b_37photos_attention_20261125/train/checkpoint-400
output_dir=outputs/agent_sft_en_27b_decode_compare_20261125

"$python_bin" -u scripts/audit_cached_teacher_logits.py \
  --model "$model_path" --adapter "$adapter" \
  --data data/agent_sft_en_27b_37photos_20261125/train.jsonl \
  --id clean_558900_1287693_blur_step_1 --tokens 192 \
  --out "$output_dir/cached_teacher_logits.json"

"$python_bin" -u scripts/eval_agent_tasks.py \
  --model "$model_path" --adapter "$adapter" \
  --tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --split val --out "$output_dir/attention_deterministic" \
  --max-steps 8 --max-new-tokens 4096 --system-profile clean_edit \
  --checkpoint-interface --strict-extent --decode-policy deterministic
