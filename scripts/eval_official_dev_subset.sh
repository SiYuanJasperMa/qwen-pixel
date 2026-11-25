#!/bin/bash
# Evaluate a model on the official MagicBrush dev subset, then compute official metrics.
# usage: eval_official_dev_subset.sh <adapter> <out> <limit> <gpu>
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
ADAPTER="$1"; OUT="$2"; LIMIT="${3:-64}"; GPU="${4:-6}"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_local_qwen_oneshot.py \
  --model "$MODEL" --adapter "$ADAPTER" \
  --tasks data/magicbrush_official_dev_v1/tasks.jsonl \
  --out "$OUT" --limit "$LIMIT" --max-new-tokens 6144 \
  > "$OUT.eval.log" 2>&1
CUDA_VISIBLE_DEVICES="$GPU" HF_ENDPOINT=https://hf-mirror.com "$PY" scripts/eval_official_metrics.py \
  --results "$OUT/results.jsonl" --tasks data/magicbrush_official_dev_v1/tasks.jsonl \
  > "$OUT.metrics.log" 2>&1
echo "official dev eval done: $OUT"
