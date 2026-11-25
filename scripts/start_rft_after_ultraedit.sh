#!/bin/bash
# Wait for the RFT selection, build the SFT dataset, then train on GPU 0 once the
# ultraedit-only run finishes.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python

while [ ! -f data/sft_magicbrush_rft_v1/selected.jsonl ]; do sleep 60; done
if [ ! -f data/sft_magicbrush_rft_oneshot_v1/train.jsonl ]; then
  "$PY" scripts/build_magicbrush_oneshot_sft.py \
    --judged data/sft_magicbrush_rft_v1/selected.jsonl \
    --out data/sft_magicbrush_rft_oneshot_v1 > outputs/rft_build.log 2>&1
fi

while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ] && break
  sleep 60
done
mkdir -p outputs/sft_magicbrush_rft_v1/train
setsid nohup env QWEN_PIXEL_TRAIN_GPU=0 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_magicbrush_rft_oneshot_v1/train.jsonl outputs/sft_magicbrush_rft_v1/train 1600 \
  > outputs/sft_magicbrush_rft_v1/train.log 2>&1 < /dev/null &
echo "RFT training launched on GPU 0"
