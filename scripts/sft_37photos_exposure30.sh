#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
condition=${1:?Specify geometry1_control or geometry4}
decode_policy=${2:-deterministic}
case "$condition" in
  geometry1_control) geometry_weight=1 ;;
  geometry4) geometry_weight=4 ;;
  *) echo 'Unknown geometry condition' >&2; exit 2 ;;
esac
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
initial_adapter=outputs/agent_sft_en_27b_37photos_${condition}_20261125/train/checkpoint-400
output_dir=outputs/agent_sft_en_27b_37photos_${condition}_exposure30_20261125
if [[ ! -f "$initial_adapter/run_state.json" ]]; then
  echo 'The matched continuation checkpoint is not saved yet.' >&2
  exit 2
fi
if [[ -e "$output_dir/train" ]]; then
  echo 'Training output exists; inspect the actual process before reusing it.' >&2
  exit 2
fi

# Original400 + matched continuation400 + this2740 =3540 optimizer steps.
# At accumulation2 and236 rows, this matches the pilot's mean30 visits/row.
"$python_bin" -u train/run_pixel_sft.py train \
  --model "$model_path" --adapter "$initial_adapter" \
  --data data/agent_sft_en_27b_37photos_20261125/train.jsonl \
  --output "$output_dir/train" --steps 2740 --accum 2 --rank 16 \
  --lr 3e-5 --lora-scope attention --gradient-checkpointing \
  --save-every 400 --eval-every 0 --max-seq-len 8192 \
  --image-max-pixels 262144 --nonzero-weight 1 --geometry-weight "$geometry_weight" \
  --agent-val-tasks data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl \
  --agent-val-every 400 --agent-val-split val --agent-val-profile clean_edit \
  --agent-val-decode-policy "$decode_policy"
