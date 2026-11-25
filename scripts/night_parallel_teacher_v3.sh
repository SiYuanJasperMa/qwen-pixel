#!/bin/bash
# Teacher-data production scheduler v3: moderate during Beijing day, high parallelism at night.
# Night: MagicBrush variants 16 + SciForma shards 1-4 (8 each, rotating teachers) +
#        UltraEdit shards 6-8 (8 each, rotating teachers).
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/night_teacher_v3
mkdir -p "$LOG"

is_night() {
  local hour
  hour=$(TZ=Asia/Shanghai date +%H)
  [ "$hour" -ge 22 ] || [ "$hour" -lt 8 ]
}

variants_done() {
  local f=data/teacher_magicbrush_variants_v1/variants.jsonl
  [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 5000 ]
}

start_variants() {
  local workers="$1"
  variants_done && return
  if pgrep -f "build_magicbrush_variants.py" >/dev/null; then
    if pgrep -f "build_magicbrush_variants.py --workers $workers" >/dev/null; then
      return
    fi
    pkill -f "build_magicbrush_variants.py"
    sleep 3
  fi
  nohup "$PY" scripts/build_magicbrush_variants.py --workers "$workers" --target 5000 \
    > "$LOG/variants.log" 2>&1 &
  echo "started variants workers=$workers $(TZ=Asia/Shanghai date)" >> "$LOG/scheduler.log"
}

start_sciforma() {
  local shard="$1" teacher="$2" workers="$3"
  local out="data/sciforma_teacher_shard${shard}_v1"
  [ -f "$out/summary.json" ] && return
  pgrep -f "build_editing_teacher_pilot.py --dataset sciforma --shard $shard " >/dev/null && return
  nohup "$PY" scripts/build_editing_teacher_pilot.py --dataset sciforma --shard "$shard" \
    --teacher "$teacher" --limit 0 --workers "$workers" --out "$out" \
    > "$LOG/sciforma_${shard}.log" 2>&1 &
  echo "started sciforma shard=$shard teacher=$teacher workers=$workers $(TZ=Asia/Shanghai date)" >> "$LOG/scheduler.log"
}

start_ultraedit() {
  local shard="$1" teacher="$2" workers="$3"
  local out="data/ultraedit_teacher_shard${shard}_v1"
  [ -f "$out/summary.json" ] && return
  pgrep -f "build_ultraedit_teacher_pilot.py --shard $shard " >/dev/null && return
  nohup "$PY" scripts/build_ultraedit_teacher_pilot.py --shard "$shard" \
    --teacher "$teacher" --limit 0 --workers "$workers" --out "$out" \
    > "$LOG/ultraedit_${shard}.log" 2>&1 &
  echo "started ultraedit shard=$shard teacher=$teacher workers=$workers $(TZ=Asia/Shanghai date)" >> "$LOG/scheduler.log"
}

while true; do
  if is_night; then
    start_variants 16
    start_sciforma 1 gpt61 8
    start_sciforma 2 deepseek 8
    start_sciforma 3 deepseek 8
    start_sciforma 4 deepseek 8
    start_sciforma 5 gpt61 6
    start_sciforma 6 deepseek 6
    start_sciforma 7 deepseek 6
    start_sciforma 8 deepseek 6
  else
    start_variants 8
  fi
  sleep 600
done
