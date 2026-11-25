#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
out=outputs/agent_sft_eos_label_regression_20261125
[[ ! -e "$out/train" ]] || exit 2
"$python_bin" -u train/run_pixel_sft.py train \
  --model "$model_path" --adapter outputs/agent_sft_en_27b_overfit_20261125/train/checkpoint-120 \
  --data data/sft_exact_replay_v1/train.jsonl --output "$out/train" \
  --steps 20 --accum 1 --rank 16 --lora-scope standard --lr 1e-5 \
  --gradient-checkpointing --save-every 10 --eval-every 0 --max-seq-len 8192 \
  --image-max-pixels 262144 --geometry-weight 1 --nonzero-weight 1 \
  --agent-val-tasks data/sft_exact_replay_v1/tasks.jsonl --agent-val-split train \
  --agent-val-limit 1 --agent-val-every 10 --agent-val-profile rgb --agent-val-decode-policy deterministic
"$python_bin" -u scripts/check_training_prefixes.py \
  --model "$model_path" --adapter "$out/train/checkpoint-20" \
  --data data/sft_exact_replay_v1/train.jsonl --out "$out/prefixes.json" \
  --limit 0 --max-new-tokens 4096 --decode-policy deterministic
