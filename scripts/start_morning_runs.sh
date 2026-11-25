#!/bin/bash
# GPUs are free again: start the main morning runs.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/resume outputs/sft_all_v12 outputs/sft_magicbrush_27b_v4_cont outputs/sft_mixed_v3_cont

# GPU 0: v12 final 10k all-data (main run, 2,400 steps)
setsid nohup env QWEN_PIXEL_TRAIN_GPU=0 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_all_v12/train.jsonl outputs/sft_all_v12/train 2400 \
  > outputs/sft_all_v12/train.log 2>&1 < /dev/null &

# GPU 1: v4 continuation (+700 from ckpt-900)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_magicbrush_27b_v4/train/checkpoint-900 \
  data/sft_magicbrush_oneshot_v4/train_filtered.jsonl \
  outputs/sft_magicbrush_27b_v4_cont/train 700 1 \
  > outputs/resume/v4_cont.log 2>&1 < /dev/null &

# GPU 7: v5 continuation (+1,250 from ckpt-750)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_mixed_v3/train/checkpoint-750 \
  data/sft_mixed_v3/train.jsonl \
  outputs/sft_mixed_v3_cont/train 1250 7 \
  > outputs/resume/v5_cont.log 2>&1 < /dev/null &

# GPU 3: RFT (best-of-N, 5,250 rows)
setsid nohup env QWEN_PIXEL_TRAIN_GPU=3 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_magicbrush_rft_oneshot_v1/train.jsonl outputs/sft_magicbrush_rft_v1/train 1600 \
  > outputs/sft_magicbrush_rft_v1/train.log 2>&1 < /dev/null &

sleep 2
echo "started v12 (GPU0), v4-cont (GPU1), v5-cont (GPU7), RFT (GPU3)"
