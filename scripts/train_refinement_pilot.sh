#!/bin/bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=6
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
task_python=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
"$task_python" -c 'import json,hashlib; from pathlib import Path; p=Path("data/sft_refinement_pilot_v1/train.jsonl"); a=json.loads((p.parent/"dataset_budget_audit.json").read_text())["summary"]; assert a["records"]==1552 and a["context_overflows"]==0 and a["dataset_sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()'
"$task_python" train/run_pixel_sft.py train \
  --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B \
  --data data/sft_refinement_pilot_v1/train.jsonl --output outputs/sft_refinement_pilot_v1/train \
  --steps 1600 --accum 2 --rank 16 --lr 5e-5 --lora-scope standard \
  --geometry-weight 3 --nonzero-weight 1 --gradient-checkpointing \
  --save-every 400 --eval-every 0 --max-seq-len 8192 --image-max-pixels 262144 \
  --checkpoint-hook scripts/clean_checkpoint_eval.py
