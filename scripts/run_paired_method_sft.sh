#!/bin/bash
# Same initial model, selected tasks, teacher identities and assistant-token budget.
# Usage: run_paired_method_sft.sh ti00|ti25|ti50 gpu snapshot
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
case "$1" in ti00|ti25|ti50) ;; *) exit 2;; esac
export CUDA_VISIBLE_DEVICES="$2"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
task_data="$3/$1/train.jsonl"
task_output="outputs/paired_method_sft_20261125/$1"
"$task_python" scripts/audit_sft_dataset_budget.py --data "$task_data" --output-ceiling 6144
"$task_python" train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --data "$task_data" --output "$task_output/train" \
  --steps 3000 --supervised-token-budget 1000000 --accum 2 --rank 16 \
  --lora-scope standard --lr 2e-5 --seed 20261125 --nonzero-weight 1 \
  --geometry-weight 1 --gradient-checkpointing --save-every 250 --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144
task_checkpoint=$("$task_python" -c 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]);print(max(p.glob("checkpoint-*"),key=lambda f:int(f.name.split("-")[-1])))' "$task_output/train")
"$task_python" scripts/eval_local_qwen_oneshot.py \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
  --adapter "$task_checkpoint" --tasks data/rl_smoke_en_20261125/tasks_eval64.jsonl \
  --out "$task_output/eval" --limit 64 --max-new-tokens 6144 --decode-policy deterministic
