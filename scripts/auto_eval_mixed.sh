#!/bin/bash
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
TRAIN=outputs/sft_mixed_oneshot_v1/train
OUT=outputs/sft_mixed_oneshot_v1
for CKPT in 200 400; do
  while [ ! -d "$TRAIN/checkpoint-$CKPT" ]; do
    if ! pgrep -f "run_pixel_sft.py train.*sft_mixed_oneshot_v1" >/dev/null; then
      echo "training exited before checkpoint-$CKPT" >&2
      exit 1
    fi
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=7 "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks data/magicbrush_dev32_v1/tasks.jsonl --out "$OUT/development_${CKPT}_magicbrush" \
    --limit 32 --max-new-tokens 6144 > "$OUT/development_${CKPT}_magicbrush.log" 2>&1
  CUDA_VISIBLE_DEVICES=7 "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks data/sft_ultraedit_oneshot_v1/dev_tasks.jsonl --out "$OUT/development_${CKPT}_ultraedit" \
    --limit 24 --max-new-tokens 6144 > "$OUT/development_${CKPT}_ultraedit.log" 2>&1
  CUDA_VISIBLE_DEVICES=7 "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks data/sft_sciforma_oneshot_v1/dev_tasks.jsonl --out "$OUT/development_${CKPT}_sciforma" \
    --limit 14 --max-new-tokens 6144 > "$OUT/development_${CKPT}_sciforma.log" 2>&1
  echo "evaluated mixed checkpoint-$CKPT"
done
echo "mixed eval done"
