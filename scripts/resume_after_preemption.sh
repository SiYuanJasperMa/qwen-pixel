#!/bin/bash
# Resume the interrupted overnight runs once GPUs on the worker free up.
# Each job waits for its GPU to have <5GB used, then launches detached.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/resume

wait_gpu() {
  local gpu="$1"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 99999)
    [ "$used" -lt 5000 ] && return
    sleep 120
  done
}

(
  wait_gpu 1
  setsid nohup bash scripts/continue_sft_generic.sh \
    outputs/sft_magicbrush_27b_v4/train/checkpoint-900 \
    data/sft_magicbrush_oneshot_v4/train_filtered.jsonl \
    outputs/sft_magicbrush_27b_v4_cont/train 700 1 \
    > outputs/resume/v4_cont.log 2>&1 < /dev/null &
) &
(
  wait_gpu 7
  setsid nohup bash scripts/continue_sft_generic.sh \
    outputs/sft_mixed_v3/train/checkpoint-750 \
    data/sft_mixed_v3/train.jsonl \
    outputs/sft_mixed_v3_cont/train 1250 7 \
    > outputs/resume/v5_cont.log 2>&1 < /dev/null &
) &
(
  wait_gpu 0
  setsid nohup env QWEN_PIXEL_TRAIN_GPU=0 bash scripts/train_magicbrush_sft_27b.sh \
    data/sft_magicbrush_rft_oneshot_v1/train.jsonl outputs/sft_magicbrush_rft_v1/train 1600 \
    > outputs/sft_magicbrush_rft_v1/train.log 2>&1 < /dev/null &
) &
(
  wait_gpu 3
  setsid nohup env QWEN_PIXEL_TRAIN_GPU=3 bash scripts/train_magicbrush_sft_27b.sh \
    data/sft_rft_ti_v1/train.jsonl outputs/sft_rft_ti_v1/train 1600 \
    > outputs/sft_rft_ti_v1/train.log 2>&1 < /dev/null &
) &
(
  wait_gpu 5
  setsid nohup bash scripts/v4_checkpoint_eval_watcher_v2.sh \
    > outputs/resume/v4_watcher.log 2>&1 < /dev/null &
) &
wait
