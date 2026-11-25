#!/bin/bash
pkill -9 -f "eval_local_qwen_oneshot.py.*sft_magicbrush_27b_v4/train/checkpoint-400"
pkill -9 -f "v4_checkpoint_eval_watcher_v2"
sleep 1
pgrep -af "eval_local_qwen_oneshot.py.*sft_magicbrush_27b_v4" | head -3
echo killed
