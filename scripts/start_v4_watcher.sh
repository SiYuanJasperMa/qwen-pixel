#!/bin/bash
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
setsid nohup bash scripts/v4_checkpoint_eval_watcher_v2.sh > outputs/v4_watcher.log 2>&1 < /dev/null &
sleep 1
