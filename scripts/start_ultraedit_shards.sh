#!/bin/bash
# Restart UltraEdit teacher shards 6-8 after fixing the PARQUET path type.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/night_teacher_v3
mkdir -p "$LOG"
setsid nohup $PY scripts/build_ultraedit_teacher_pilot.py --shard 6 --teacher gpt61 --limit 0 --workers 8 \
  --out data/ultraedit_teacher_shard6_v1 > "$LOG/ultraedit_6.log" 2>&1 < /dev/null &
setsid nohup $PY scripts/build_ultraedit_teacher_pilot.py --shard 7 --teacher gpt55 --limit 0 --workers 8 \
  --out data/ultraedit_teacher_shard7_v1 > "$LOG/ultraedit_7.log" 2>&1 < /dev/null &
setsid nohup $PY scripts/build_ultraedit_teacher_pilot.py --shard 8 --teacher luna --limit 0 --workers 8 \
  --out data/ultraedit_teacher_shard8_v1 > "$LOG/ultraedit_8.log" 2>&1 < /dev/null &
sleep 2
