#!/bin/bash
# Build the next TI dataset round after generators go quiet (or timeout),
# then write a READY marker for the training watcher on the GPU worker.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/auto_next_round2.log
NEXT=${1:-v15}
MAX_MIN=${2:-180}

echo "[$(date)] builder2 started for $NEXT" >> "$LOG"
for i in $(seq 1 "$MAX_MIN"); do
  n=$(pgrep -af "^[^ ]*python .*build_magicbrush_target_informed.py" | grep -v seed21pro | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[$(date)] generators finished" >> "$LOG"
    break
  fi
  sleep 60
done
echo "[$(date)] building TI v8" >> "$LOG"
$PY scripts/build_target_informed_sft.py \
  --dirs data/teacher_target_informed_deepseek,data/teacher_target_informed_gpt55,data/teacher_target_informed_gpt61_mask,data/teacher_target_informed_luna,data/teacher_target_informed_gpt56sol,data/teacher_target_informed_seed21pro \
  --out data/sft_target_informed_v8 >> "$LOG" 2>&1
echo "[$(date)] merging $NEXT" >> "$LOG"
$PY scripts/merge_sft_datasets.py --inputs data/sft_all_v12,data/sft_target_informed_v8 \
  --out "data/sft_all_$NEXT" --no-replace >> "$LOG" 2>&1
touch "data/sft_all_$NEXT/READY"
echo "[$(date)] READY written for $NEXT" >> "$LOG"
