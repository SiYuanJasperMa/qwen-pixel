#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
root=outputs/agent_sft_exact_continuation_20261125
common=(--model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
  --data data/sft_exact_replay_v1/train.jsonl --steps 12 --accum 1 --rank 16
  --lora-scope standard --lr 1e-5 --seed 42 --gradient-checkpointing
  --save-every 4 --keep-checkpoint-training-state --eval-every 0
  --max-seq-len 8192 --image-max-pixels 262144 --geometry-weight 1 --nonzero-weight 1
  --agent-val-tasks data/sft_exact_replay_v1/tasks.jsonl --agent-val-split train
  --agent-val-limit 1 --agent-val-every 4 --agent-val-profile rgb --agent-val-decode-policy deterministic)
"$python_bin" -u train/run_pixel_sft.py train "${common[@]}" \
  --adapter outputs/agent_sft_en_27b_overfit_20261125/train/checkpoint-120 --output "$root/uninterrupted"
"$python_bin" -u train/run_pixel_sft.py train "${common[@]}" \
  --adapter outputs/agent_sft_en_27b_overfit_20261125/train/checkpoint-120 \
  --output "$root/interrupted" --stop-after-step 4
"$python_bin" -u train/run_pixel_sft.py train "${common[@]}" \
  --resume-training-state "$root/interrupted/checkpoint-4/training_state.pt" --output "$root/resumed"
"$python_bin" -u scripts/check_training_continuation_results.py --root "$root"
