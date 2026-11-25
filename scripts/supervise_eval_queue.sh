#!/bin/bash
# Keep an eval_after_training queue alive.
# usage: supervise_eval_queue.sh <train_dir> <out_dir> <gpu> <ckpts>
set -u
TRAIN="$1"; OUT="$2"; GPU="$3"; CKPTS="$4"
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
LOG="$OUT/eval_supervisor.log"
while true; do
  n=$(pgrep -fc "^bash scripts/eval_after_training.sh $TRAIN" || true)
  if [ "$n" -eq 0 ]; then
    echo "[$(date)] queue dead, restarting" >> "$LOG"
    setsid nohup bash scripts/eval_after_training.sh "$TRAIN" "$OUT" "$GPU" "$CKPTS" >> "$OUT/eval_queue.log" 2>&1 < /dev/null &
  fi
  sleep 300
done
