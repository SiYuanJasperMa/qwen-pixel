#!/bin/bash
# Queues that start when a GPU frees up:
#  - GPU 6: evaluate mixed v2 checkpoints 200/400/600/800 on MagicBrush dev32
#  - GPU 1: continue v4 for +800 steps from checkpoint-1600
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ]
}

(
  while ! gpu_free 6; do sleep 60; done
  for CKPT in 200 400 600 800; do
    [ -d "outputs/sft_mixed_v2/train/checkpoint-$CKPT" ] || continue
    [ -f "outputs/sft_mixed_v2/development_${CKPT}_magicbrush/summary.json" ] && continue
    CUDA_VISIBLE_DEVICES=6 "$PY" scripts/eval_local_qwen_oneshot.py \
      --model "$MODEL" --adapter "outputs/sft_mixed_v2/train/checkpoint-$CKPT" \
      --tasks data/magicbrush_dev32_v1/tasks.jsonl \
      --out "outputs/sft_mixed_v2/development_${CKPT}_magicbrush" \
      --limit 32 --max-new-tokens 6144 > "outputs/sft_mixed_v2/development_${CKPT}_magicbrush.log" 2>&1
    echo "evaluated mixed v2 checkpoint-$CKPT"
  done
) &
(
  while :; do
    if [ -d outputs/sft_magicbrush_27b_v4/train/checkpoint-1600 ] && gpu_free 1; then
      break
    fi
    sleep 60
  done
  mkdir -p outputs/sft_magicbrush_27b_v4_cont
  setsid nohup env QWEN_PIXEL_TRAIN_GPU=1 bash scripts/continue_27b_v4.sh 800 \
    > outputs/sft_magicbrush_27b_v4_cont/train.log 2>&1 < /dev/null &
  echo "v4 continuation launched"
) &
sleep 2
