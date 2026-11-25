#!/bin/bash
# Launch the remaining two SFT runs on unfiltered splits (trainer skips over-long rows).
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/sft_ultraedit_27b
setsid nohup env QWEN_PIXEL_TRAIN_GPU=0 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_ultraedit_oneshot_full_v1/train.jsonl outputs/sft_ultraedit_27b/train 800 \
  > outputs/sft_ultraedit_27b/train.log 2>&1 < /dev/null &
setsid nohup bash scripts/wait_gpu_then_train.sh 3 data/sft_mixed_v6_lowratio/train.jsonl \
  outputs/sft_mixed_v6_lowratio/train 2000 > outputs/gpu3_queue.log 2>&1 < /dev/null &
sleep 2
