#!/bin/bash
# Restart eval watchers with the faster 16-task / 4096-token configuration.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
pkill -9 -f "v4_checkpoint_eval_watcher_v2"
pkill -9 -f "eval_after_training.sh"
sleep 2
setsid nohup bash scripts/v4_checkpoint_eval_watcher_v2.sh > outputs/v4_watcher.log 2>&1 < /dev/null &
bash scripts/start_eval_queues.sh
sleep 2
