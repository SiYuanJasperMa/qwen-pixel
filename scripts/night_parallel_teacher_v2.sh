#!/bin/bash
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/night_teacher
mkdir -p "$LOG"
while true; do
  hour=$(TZ=Asia/Shanghai date +%H)
  if [ "$hour" -ge 22 ] || [ "$hour" -lt 8 ]; then
    echo "night window $(TZ=Asia/Shanghai date) starting high-parallel teachers" >> "$LOG/scheduler.log"
    pkill -f "build_ultraedit_teacher_pilot.py" 2>/dev/null
    pkill -f "build_editing_teacher_pilot.py --dataset sciforma" 2>/dev/null
    pkill -f "build_magicbrush_variants.py" 2>/dev/null
    sleep 3
    nohup "$PY" scripts/build_ultraedit_teacher_pilot.py --limit 0 --workers 16 \
      --out data/ultraedit_teacher_pilot_v1 > "$LOG/ultraedit.log" 2>&1 &
    for shard in 1 2 3 4; do
      nohup "$PY" scripts/build_editing_teacher_pilot.py --dataset sciforma --shard "$shard" --limit 0 --workers 8 \
        --out "data/sciforma_teacher_shard${shard}_v1" > "$LOG/sciforma_${shard}.log" 2>&1 &
    done
    nohup "$PY" scripts/build_magicbrush_variants.py --workers 16 --target 5000 \
      > "$LOG/variants.log" 2>&1 &
    echo "started ultraedit=16 sciforma_shards1-4=8each variants=16" >> "$LOG/scheduler.log"
    break
  fi
  sleep 300
done
