#!/bin/bash
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
mkdir -p outputs/star_pilot
setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher deepseek --limit 1000 --workers 8 \
  --out data/teacher_target_informed_deepseek \
  > outputs/star_pilot/deepseek.log 2>&1 < /dev/null &
sleep 2
