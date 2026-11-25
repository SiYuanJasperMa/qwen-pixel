#!/bin/bash
# Relaunch the GPU 7 v5 queue detached.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
setsid nohup bash scripts/wait_gpu_then_train.sh 7 data/sft_mixed_v3/train_filtered.jsonl outputs/sft_mixed_v3/train 2000 > outputs/gpu7_v5_queue.log 2>&1 < /dev/null &
sleep 1
