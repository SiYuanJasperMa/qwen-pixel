#!/bin/bash
# Recompute official metrics (now including DINO) on CPU for existing results.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
export CUDA_VISIBLE_DEVICES=""
export HF_ENDPOINT=https://hf-mirror.com
$PY scripts/eval_official_metrics.py \
  --results outputs/sft_magicbrush_27b_oneshot_v3_cont/development_1200/results.jsonl \
  --tasks data/magicbrush_dev32_v1/tasks.jsonl > outputs/official_metrics_v3cont.log 2>&1
$PY scripts/eval_official_metrics.py \
  --results data/teacher_eval_gpt55/judged.jsonl \
  --tasks data/magicbrush_dev32_v1/tasks.jsonl > outputs/official_metrics_gpt55.log 2>&1
$PY scripts/eval_official_metrics.py \
  --results data/teacher_eval_gpt61/judged.jsonl \
  --tasks data/magicbrush_dev32_v1/tasks.jsonl > outputs/official_metrics_gpt61.log 2>&1
$PY scripts/eval_official_metrics.py \
  --results outputs/crossbench_v3cont1200_sciforma/results.jsonl \
  --tasks data/sft_sciforma_oneshot_full_v1/dev_tasks.jsonl > outputs/official_metrics_v3cont_sciforma.log 2>&1
$PY scripts/eval_official_metrics.py \
  --results outputs/crossbench_zero_sciforma/results.jsonl \
  --tasks data/sft_sciforma_oneshot_full_v1/dev_tasks.jsonl > outputs/official_metrics_zero_sciforma.log 2>&1
echo done
