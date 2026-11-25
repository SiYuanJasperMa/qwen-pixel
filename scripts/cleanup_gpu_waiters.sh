#!/bin/bash
# GPU was reclaimed for videogen; remove all GPU-waiting/auto-restart scripts so
# nothing starts training when the GPUs free. Data production (CPU/API) stays.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
pkill -9 -f "resume_after_preemption.sh"
pkill -9 -f "start_v9_rft_ti.sh"
pkill -9 -f "start_v4_continuation_queue.sh"
pkill -9 -f "eval_after_training.sh"
pkill -9 -f "wait_gpu_then_train.sh"
pkill -9 -f "v4_checkpoint_eval_watcher_v2.sh"
pkill -9 -f "start_eval_queues.sh"
pkill -9 -f "start_after_queues.sh"
pkill -9 -f "start_rft_after_ultraedit.sh"
pkill -9 -f "start_ultraedit_and_v6.sh"
sleep 2
echo "remaining gpu-waiting:"
pgrep -af "resume_after_preemption|start_v9_rft_ti|eval_after_training|wait_gpu_then_train|v4_checkpoint_eval|start_eval_queues" | grep -v pgrep | head -5
echo done
