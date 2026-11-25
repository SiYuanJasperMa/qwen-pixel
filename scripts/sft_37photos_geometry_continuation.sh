#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
condition=${1:?Specify geometry1_control or geometry4}
case "$condition" in
  geometry1_control) geometry_weight=1 ;;
  geometry4) geometry_weight=4 ;;
  *) echo 'Unknown geometry condition' >&2; exit 2 ;;
esac
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
initial_adapter=outputs/agent_sft_en_27b_37photos_attention_20261125/train/checkpoint-400
data_path=data/agent_sft_en_27b_37photos_20261125/train.jsonl
output_dir=outputs/agent_sft_en_27b_37photos_${condition}_20261125

"$python_bin" -u train/run_pixel_sft.py train \
  --model "$model_path" --adapter "$initial_adapter" \
  --data "$data_path" --output "$output_dir/train" \
  --steps 400 --accum 2 --rank 16 --lr 5e-5 --lora-scope attention \
  --gradient-checkpointing --save-every 400 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1 \
  --geometry-weight "$geometry_weight"

"$python_bin" -u scripts/eval_agent_tasks.py \
  --model "$model_path" --adapter "$output_dir/train/checkpoint-400" \
  --tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --split val --out "$output_dir/development_400" \
  --max-steps 8 --max-new-tokens 4096 \
  --system-profile clean_edit --checkpoint-interface --strict-extent

"$python_bin" -u scripts/check_training_prefixes.py \
  --model "$model_path" --adapter "$output_dir/train/checkpoint-400" \
  --data "$data_path" --out "$output_dir/prefixes12.json" \
  --limit 12 --max-new-tokens 4096
