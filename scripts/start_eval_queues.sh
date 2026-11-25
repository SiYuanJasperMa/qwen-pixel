#!/bin/bash
# Queue post-training checkpoint evals for v5, v6 and RFT on their own GPUs.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
setsid nohup bash scripts/eval_after_training.sh \
  outputs/sft_mixed_v3/train outputs/sft_mixed_v3 7 1000,2000 \
  > outputs/eval_queue_v5.log 2>&1 < /dev/null &
setsid nohup bash scripts/eval_after_training.sh \
  outputs/sft_mixed_v6_lowratio/train outputs/sft_mixed_v6_lowratio 3 1000,2000 \
  > outputs/eval_queue_v6.log 2>&1 < /dev/null &
setsid nohup bash scripts/eval_after_training.sh \
  outputs/sft_magicbrush_rft_v1/train outputs/sft_magicbrush_rft_v1 0 400,800,1200,1600 \
  > outputs/eval_queue_rft.log 2>&1 < /dev/null &
sleep 2
