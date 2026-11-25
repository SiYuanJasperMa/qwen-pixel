#!/bin/bash
# Wait for SFT checkpoints, generate code on val set, then execute+judge.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel || exit 1
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B
OUT=outputs/sft_code_coco_v1
DATA=data/sft_code_coco_v1/val.jsonl
GPU=${EVAL_GPU:-1}

# Base-model baseline first (no adapter).
if [ ! -f outputs/gen_sft_v1_base.jsonl ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY train/run_pixel_sft.py generate \
    --model "$MODEL" --data "$DATA" \
    --output outputs/gen_sft_v1_base.jsonl \
    --limit 20 --max-new-tokens 600 > outputs/gen_sft_v1_base.log 2>&1
  $PY eval/eval_sft_coco_code.py --generations outputs/gen_sft_v1_base.jsonl \
    --data "$DATA" --out outputs/eval_sft_v1_base > outputs/eval_sft_v1_base.log 2>&1
fi

for CKPT in 100 300 500; do
  while [ ! -d "$OUT/checkpoint-$CKPT" ]; do
    if ! pgrep -f "run_pixel_sft.py train" > /dev/null; then
      echo "training exited before checkpoint-$CKPT" >&2
      exit 1
    fi
    sleep 20
  done
  GEN=outputs/gen_sft_v1_ckpt$CKPT.jsonl
  if [ ! -f "$GEN" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY train/run_pixel_sft.py generate \
      --model "$MODEL" --adapter "$OUT/checkpoint-$CKPT" --data "$DATA" \
      --output "$GEN" --limit 20 --max-new-tokens 1200 > outputs/gen_sft_v1_ckpt$CKPT.log 2>&1
  fi
  $PY eval/eval_sft_coco_code.py --generations "$GEN" \
    --data "$DATA" --out outputs/eval_sft_v1_ckpt$CKPT > outputs/eval_sft_v1_ckpt$CKPT.log 2>&1
done
echo ALL_EVALS_DONE
