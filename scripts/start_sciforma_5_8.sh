#!/bin/bash
# Start SciForma teacher shards 5-8 (newly downloaded) with rotating teachers.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/night_teacher_v3
mkdir -p "$LOG"
start() {
  local shard="$1" teacher="$2"
  local out="data/sciforma_teacher_shard${shard}_v1"
  [ -f "$out/summary.json" ] && return
  pgrep -f "build_editing_teacher_pilot.py --dataset sciforma --shard $shard " >/dev/null && return
  setsid nohup $PY scripts/build_editing_teacher_pilot.py --dataset sciforma --shard "$shard" \
    --teacher "$teacher" --limit 0 --workers 6 --out "$out" \
    > "$LOG/sciforma_${shard}.log" 2>&1 < /dev/null &
  echo "started sciforma shard=$shard teacher=$teacher"
}
start 5 gpt61
start 6 luna
start 7 seed21pro
start 8 gpt55
sleep 2
