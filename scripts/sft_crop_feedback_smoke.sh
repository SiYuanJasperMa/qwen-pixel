#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
condition=${1:?Specify paired or positive_control}
case "$condition" in
  paired) train_data=data/sft_crop_correction_20261125_worker/train.jsonl ;;
  positive_control) train_data=data/sft_crop_positive_control_20261125/train.jsonl ;;
  *) echo 'Unknown crop-feedback condition' >&2; exit 2 ;;
esac
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
initial_adapter=outputs/agent_sft_en_27b_37photos_geometry4_20261125/train/checkpoint-400
output_dir=outputs/agent_sft_crop_feedback_${condition}_20261125
if [[ -e "$output_dir/train" ]]; then
  echo 'Training output exists; inspect the actual process before any restart.' >&2
  exit 2
fi

"$python_bin" -u train/run_pixel_sft.py train \
  --model "$model_path" --adapter "$initial_adapter" --data "$train_data" \
  --output "$output_dir/train" --steps 480 --accum 2 --rank 16 --lr 2e-5 \
  --lora-scope attention --geometry-weight 1 --nonzero-weight 1 \
  --gradient-checkpointing --save-every 240 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144 \
  --agent-val-tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --agent-val-profile clean_edit --agent-val-every 240 --agent-val-decode-policy deterministic

"$python_bin" -u scripts/check_training_prefixes.py \
  --model "$model_path" --adapter "$output_dir/train/checkpoint-480" \
  --data outputs/agent_sft_en_27b_crop_recovery_probe_20261125/prefixes.jsonl \
  --out "$output_dir/crop_probe_predictions.json" --limit 0 \
  --max-new-tokens 4096 --decode-policy deterministic
