#!/bin/bash
# Scale target-informed (STaR) generation to the full MagicBrush train set with 4 teachers.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
mkdir -p outputs/star_pilot

setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt55 --limit 0 --workers 8 \
  --out data/teacher_target_informed_gpt55 \
  > outputs/star_pilot/gpt55_full.log 2>&1 < /dev/null &

setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher deepseek --limit 0 --workers 8 \
  --out data/teacher_target_informed_deepseek \
  > outputs/star_pilot/deepseek_full.log 2>&1 < /dev/null &

setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt61 --limit 0 --workers 6 --use-mask \
  --out data/teacher_target_informed_gpt61_mask \
  > outputs/star_pilot/gpt61_full.log 2>&1 < /dev/null &

setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt56sol --limit 0 --workers 8 \
  --out data/teacher_target_informed_gpt56sol \
  > outputs/star_pilot/gpt56sol_full.log 2>&1 < /dev/null &

sleep 2
echo "started TI full: gpt55=8 deepseek=8 gpt61_mask=6 gpt56sol=8"
