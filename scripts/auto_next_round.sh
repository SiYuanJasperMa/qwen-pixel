#!/bin/bash
# Wait for target-informed generation to finish, then build TI v7 + v14 and
# launch the next training round on the 96GB worker. Runs detached on 0617.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/auto_next_round.log
W2=19ebb06cd438f7ff.worker_0.job.masiyuan-jasper@merlin-ssh-proxy-cn.byted.org
W3=5e5bed85e489ddcb.worker_0.job.masiyuan-jasper@merlin-ssh-proxy-cn.byted.org
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -p 22222"

echo "[$(date)] watcher started" >> "$LOG"
for i in $(seq 1 150); do
  n=$(pgrep -af "^[^ ]*python .*build_magicbrush_target_informed.py" | grep -v seed21pro | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[$(date)] generators finished" >> "$LOG"
    break
  fi
  sleep 60
done
echo "[$(date)] building TI v7" >> "$LOG"
$PY scripts/build_target_informed_sft.py \
  --dirs data/teacher_target_informed_deepseek,data/teacher_target_informed_gpt55,data/teacher_target_informed_gpt61_mask,data/teacher_target_informed_luna,data/teacher_target_informed_gpt56sol \
  --out data/sft_target_informed_v7 >> "$LOG" 2>&1
$PY scripts/merge_sft_datasets.py --inputs data/sft_all_v12,data/sft_target_informed_v7 \
  --out data/sft_all_v14 --no-replace >> "$LOG" 2>&1

echo "[$(date)] launching v14 fresh training on worker2 gpu6" >> "$LOG"
$SSH "$W2" "cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel && mkdir -p outputs/sft_all_v14 && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True QWEN_PIXEL_TRAIN_GPU=6 bash scripts/train_magicbrush_sft_27b.sh data/sft_all_v14/train.jsonl outputs/sft_all_v14/train 2400 /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B 100 > outputs/sft_all_v14/train.log 2>&1 < /dev/null &" >> "$LOG" 2>&1
echo "[$(date)] queueing v14 evals on worker3 gpu5" >> "$LOG"
$SSH "$W3" "cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel && setsid nohup bash scripts/eval_after_training.sh outputs/sft_all_v14/train outputs/sft_all_v14 5 600,1200,1800,2400 > outputs/sft_all_v14/eval_queue.log 2>&1 < /dev/null &" >> "$LOG" 2>&1
echo "[$(date)] watcher done" >> "$LOG"
