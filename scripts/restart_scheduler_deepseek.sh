#!/bin/bash
# Restart the night scheduler and switch SciForma shards 2/3/6/7 from
# Luna/Seed to DeepSeek (SciForma teacher eval: DeepSeek 60% vs Luna 20%).
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
kill -9 1112125 2>/dev/null
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 2 "
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 3 "
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 6 "
pkill -9 -f "build_editing_teacher_pilot.py --dataset sciforma --shard 7 "
sleep 2
setsid nohup bash scripts/night_parallel_teacher_v3.sh > outputs/night_teacher_v3.out 2>&1 < /dev/null &
sleep 2
