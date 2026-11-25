#!/bin/bash
# Start long-running queue scripts detached.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
setsid nohup bash scripts/v4_checkpoint_eval_watcher.sh > outputs/v4_watcher.log 2>&1 < /dev/null &
setsid nohup bash scripts/gpu7_queue_mixedv2_eval_then_v5.sh > outputs/gpu7_queue.log 2>&1 < /dev/null &
sleep 1
