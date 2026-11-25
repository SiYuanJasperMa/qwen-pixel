#!/bin/bash
# Scale target-informed (STaR) generation: gpt55 to 1000 now, gpt61+mask to 500
# after its 100-sample pilot exits.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt55 --limit 1000 --workers 8 \
  --out data/teacher_target_informed_gpt55 > outputs/star_pilot/gpt55_scale.log 2>&1 < /dev/null &

while pgrep -f "build_magicbrush_target_informed.py --teacher gpt61 " >/dev/null; do sleep 120; done
setsid nohup $PY scripts/build_magicbrush_target_informed.py \
  --teacher gpt61 --limit 500 --workers 4 --use-mask \
  --out data/teacher_target_informed_gpt61_mask > outputs/star_pilot/gpt61_scale.log 2>&1 < /dev/null &
sleep 2
