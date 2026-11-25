#!/bin/bash
# Start v5 mixed-all SFT on GPU 7 from the unfiltered train split (over-long rows are
# skipped by the trainer). The filtered split is produced in parallel for bookkeeping.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/sft_mixed_v3
setsid nohup env QWEN_PIXEL_TRAIN_GPU=7 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_mixed_v3/train.jsonl outputs/sft_mixed_v3/train 2000 \
  > outputs/sft_mixed_v3/train.log 2>&1 < /dev/null &
sleep 2
