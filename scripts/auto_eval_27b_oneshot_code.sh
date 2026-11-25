#!/bin/bash
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
TRAIN=outputs/sft_magicbrush_27b_oneshot_code_v1/train
OUT=outputs/sft_magicbrush_27b_oneshot_code_v1
TASKS=data/magicbrush_dev32_v1/tasks.jsonl
for CKPT in 200 400; do
  while [ ! -d "$TRAIN/checkpoint-$CKPT" ]; do
    if ! pgrep -f "run_pixel_sft.py train.*sft_magicbrush_27b_oneshot_code_v1" >/dev/null; then
      echo "training exited before checkpoint-$CKPT" >&2
      exit 1
    fi
    sleep 60
  done
  mkdir -p "$OUT/development_$CKPT"
  CUDA_VISIBLE_DEVICES=7 "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks "$TASKS" --out "$OUT/development_$CKPT" --limit 32 \
    --max-new-tokens 6144 --prompt-mode code > "$OUT/development_$CKPT.log" 2>&1
  echo "evaluated code-only checkpoint-$CKPT"
done
echo "auto eval done"
