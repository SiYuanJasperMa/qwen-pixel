#!/bin/bash
# Wait for a training run to produce its first checkpoint, then start the eval queue.
# usage: watch_train_and_eval.sh <train_dir> <out_dir> <gpu> <ckpts>
set -u
TRAIN="$1"; OUT="$2"; GPU="$3"; CKPTS="$4"
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
LOG="$OUT/watch_eval.log"
mkdir -p "$OUT"
while [ -z "$(ls -d "$TRAIN"/checkpoint-* 2>/dev/null)" ]; do sleep 300; done
echo "[$(date)] checkpoints found, starting eval queue" >> "$LOG"
setsid nohup bash scripts/eval_after_training.sh "$TRAIN" "$OUT" "$GPU" "$CKPTS" >> "$OUT/eval_queue.log" 2>&1 < /dev/null &
setsid nohup bash scripts/supervise_eval_queue.sh "$TRAIN" "$OUT" "$GPU" "$CKPTS" > /dev/null 2>&1 < /dev/null &
