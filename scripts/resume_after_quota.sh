#!/bin/bash
# Resume the four SFT runs after the disk-quota crash. Quota has been freed.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/sft_all_v12_cont outputs/sft_magicbrush_27b_v4_cont outputs/sft_mixed_v3_cont outputs/sft_magicbrush_rft_v1_cont outputs/resume

# GPU 0: v12 continue from checkpoint-50 (2350 more steps)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_all_v12/train/checkpoint-50 \
  data/sft_all_v12/train.jsonl \
  outputs/sft_all_v12_cont/train 2350 0 \
  > outputs/resume/v12_cont.log 2>&1 < /dev/null &

# GPU 1: v4 continue from checkpoint-900 (700 more steps)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_magicbrush_27b_v4/train/checkpoint-900 \
  data/sft_magicbrush_oneshot_v4/train_filtered.jsonl \
  outputs/sft_magicbrush_27b_v4_cont/train 700 1 \
  > outputs/resume/v4_cont.log 2>&1 < /dev/null &

# GPU 7: v5 continue from checkpoint-750 (1250 more steps)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_mixed_v3/train/checkpoint-750 \
  data/sft_mixed_v3/train.jsonl \
  outputs/sft_mixed_v3_cont/train 1250 7 \
  > outputs/resume/v5_cont.log 2>&1 < /dev/null &

# GPU 3: RFT continue from checkpoint-50 (1550 more steps)
setsid nohup bash scripts/continue_sft_generic.sh \
  outputs/sft_magicbrush_rft_v1/train/checkpoint-50 \
  data/sft_magicbrush_rft_oneshot_v1/train.jsonl \
  outputs/sft_magicbrush_rft_v1_cont/train 1550 3 \
  > outputs/resume/rft_cont.log 2>&1 < /dev/null &

sleep 2
echo "resumed v12(GPU0), v4(GPU1), v5(GPU7), RFT(GPU3)"
