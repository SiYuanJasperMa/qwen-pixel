#!/bin/bash
# GPU 7 queue: wait for the mixed-v1 eval to finish, evaluate mixed v2 ckpt-200 on
# MagicBrush dev32, then start v5 mixed-all SFT.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B

while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ] && break
  sleep 60
done

mkdir -p outputs/sft_mixed_v2
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/eval_local_qwen_oneshot.py \
  --model "$MODEL" --adapter outputs/sft_mixed_v2/train/checkpoint-200 \
  --tasks data/magicbrush_dev32_v1/tasks.jsonl \
  --out outputs/sft_mixed_v2/development_200_magicbrush \
  --limit 32 --max-new-tokens 6144 > outputs/sft_mixed_v2/development_200_magicbrush.log 2>&1

while [ ! -f data/sft_mixed_v3/train_filtered.jsonl ]; do sleep 60; done
mkdir -p outputs/sft_mixed_v3
QWEN_PIXEL_TRAIN_GPU=7 nohup bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_mixed_v3/train_filtered.jsonl outputs/sft_mixed_v3/train 2000 \
  > outputs/sft_mixed_v3/train.log 2>&1 &
echo "v5 mixed-all training launched on GPU 7"
