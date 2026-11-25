#!/bin/bash
# Restart the scheduler and switch SciForma shards 4/8 from GPT-5.5 to DeepSeek so
# the Azure quota is freed for the gpt-5.5 judge used by student evals.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 4 "
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 8 "
kill -9 $(pgrep -f "bash scripts/night_parallel_teacher_v3.sh" | head -1) 2>/dev/null
sleep 2
setsid nohup bash scripts/night_parallel_teacher_v3.sh > outputs/night_teacher_v3.out 2>&1 < /dev/null &
sleep 2
