#!/bin/bash
# Watch for GPU availability and resume the interrupted Qwen SFT when a GPU frees.
# - single instance via lock dir
# - only launches when no training is already running
# - v4 MagicBrush continuation from checkpoint-900 (+700 steps to 1600)
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
mkdir -p outputs/resume
LOG=outputs/resume/gpu_watch.log

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# single instance via flock (survives PID-namespace differences)
exec 9>outputs/resume/.gpu_watch.flock
if ! flock -n 9; then
  exit 0
fi

CKPT=outputs/sft_magicbrush_27b_v4/train/checkpoint-900
DATA=data/sft_magicbrush_oneshot_v4/train_filtered.jsonl
OUT=outputs/sft_magicbrush_27b_v4_cont/train
TRAIN_LOG=outputs/resume/v4_cont.log

training_recent() {
  # A training is considered running if its process is visible or its log was
  # written in the last 10 minutes (covers other PID namespaces).
  pgrep -f "run_pixel_sft.py train" >/dev/null 2>&1 && return 0
  [ -f "$TRAIN_LOG" ] && [ -n "$(find "$TRAIN_LOG" -mmin -10 2>/dev/null)" ] && return 0
  return 1
}

log "watcher start pid=$$ (ckpt=$CKPT)"
last_state=""
tick=0
while :; do
  tick=$((tick + 1))
  raw=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$raw" ]; then
    state="no-gpu-visible (NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset})"
  else
    state=$(echo "$raw" | tr '\n' ';')
    if training_recent; then
      state="$state | training-running"
    else
      free=$(echo "$raw" | awk -F', ' '$2 < 5000 {print $1; exit}')
      if [ -n "$free" ]; then
        # second look after a cooldown to avoid racing another watcher
        sleep 180
        used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$free" 2>/dev/null || echo 99999)
        if [ -n "$used2" ] && [ "$used2" -lt 5000 ] && ! training_recent; then
          log "free GPU $free confirmed -> launching v4 continuation (ckpt-900, 700 steps)"
          setsid nohup bash scripts/continue_sft_generic.sh \
            "$CKPT" "$DATA" "$OUT" 700 "$free" \
            >> "$TRAIN_LOG" 2>&1 < /dev/null &
          sleep 30
          state="$state | launched-v4-on-$free"
        else
          state="$state | gpu-$free-became-busy"
        fi
      fi
    fi
  fi
  if [ "$state" != "$last_state" ]; then
    log "state: $state"
    last_state="$state"
  fi
  if [ $((tick % 30)) -eq 0 ]; then
    log "heartbeat tick=$tick state=$state"
  fi
  sleep 120
done
