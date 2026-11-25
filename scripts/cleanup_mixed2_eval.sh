#!/bin/bash
# Kill the slow mixed-v2 eval loop and the old after_queues wrappers, then relaunch
# only the v4 continuation waiter.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
pkill -9 -f "eval_local_qwen_oneshot.py.*sft_mixed_v2" 2>/dev/null
pkill -9 -f "start_after_queues.sh" 2>/dev/null
sleep 2
setsid nohup bash scripts/start_v4_continuation_queue.sh > outputs/v4_cont_queue.log 2>&1 < /dev/null &
sleep 2
