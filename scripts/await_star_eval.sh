#!/bin/bash
# Follow the actual training process; do not evaluate a partial checkpoint.
# usage: await_star_eval.sh pid train_dir eval_dir gpu
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p "$3"
tail --pid="$1" -n 0 -f "$(dirname "$2")/train.log" > /dev/null
exec /mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python \
  scripts/eval_completed_star_run.py --train "$2" --out "$3" --gpu "$4"
