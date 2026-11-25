#!/bin/bash
# Relaunch target-informed (STaR) pilots with the fixed prompt.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
mkdir -p outputs/star_pilot
rm -f data/teacher_target_informed_gpt61_mask/judged.jsonl
rm -f data/teacher_target_informed_gpt55/judged.jsonl
setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt61 --limit 100 --workers 4 --use-mask \
  --out data/teacher_target_informed_gpt61_mask > outputs/star_pilot/gpt61_mask.log 2>&1 < /dev/null &
setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt55 --limit 100 --workers 8 \
  --out data/teacher_target_informed_gpt55 > outputs/star_pilot/gpt55.log 2>&1 < /dev/null &
sleep 2
