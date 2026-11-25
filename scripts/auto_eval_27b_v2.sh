#!/bin/bash
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
TRAIN=outputs/sft_magicbrush_27b_v2/train
OUT=outputs/sft_magicbrush_27b_v2
TASKS=data/magicbrush_dev8_v1/tasks.jsonl
for CKPT in 200 400; do
  while [ ! -d "$TRAIN/checkpoint-$CKPT" ]; do
    if ! pgrep -f "run_pixel_sft.py train.*sft_magicbrush_27b_v2" >/dev/null; then
      echo "training exited before checkpoint-$CKPT" >&2
      exit 1
    fi
    sleep 60
  done
  if [ "$CKPT" = "200" ]; then GPU=2; else GPU=3; fi
  mkdir -p "$OUT/development_$CKPT"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/eval_agent_tasks.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks "$TASKS" --out "$OUT/development_$CKPT" --split val --limit 8 \
    --max-steps 8 --max-new-tokens 1800 --system-profile rgb \
    > "$OUT/development_$CKPT.log" 2>&1
  echo "evaluated checkpoint-$CKPT"
done
echo "auto eval done"
