#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
cohort=${1:?Specify single or three}
output_dir=${2:?Specify a fresh output directory}
case "$cohort" in
  single) data_dir=data/sft_exact_replay_v1; steps=120; learning_rate=2e-4 ;;
  three) data_dir=data/sft_three_tasks_v1; steps=300; learning_rate=1e-4 ;;
  *) echo 'Unknown correctness cohort' >&2; exit 2 ;;
esac
if [[ -e "$output_dir" ]]; then
  echo 'Output directory exists; preserve previous evidence and choose a fresh directory.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
model_path=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B

"$python_bin" -u train/run_pixel_sft.py train \
  --model "$model_path" --data "$data_dir/train.jsonl" --output "$output_dir/train" \
  --steps "$steps" --accum 1 --rank 16 --lr "$learning_rate" --lora-scope standard \
  --gradient-checkpointing --save-every "$steps" --eval-every 0 \
  --max-seq-len 8192 --image-max-pixels 262144 --nonzero-weight 1

"$python_bin" -u scripts/check_training_prefixes.py \
  --model "$model_path" --adapter "$output_dir/train/checkpoint-$steps" \
  --data "$data_dir/train.jsonl" --out "$output_dir/prefixes.json" \
  --limit 0 --max-new-tokens 4096

"$python_bin" -u scripts/eval_agent_tasks.py \
  --model "$model_path" --adapter "$output_dir/train/checkpoint-$steps" \
  --tasks "$data_dir/tasks.jsonl" --split train --out "$output_dir/train_rollouts" \
  --max-steps 8 --max-new-tokens 4096 --system-profile rgb \
  --checkpoint-interface --strict-extent

if [[ "$cohort" == three ]]; then
  "$python_bin" -u scripts/eval_agent_tasks.py \
    --model "$model_path" --adapter "$output_dir/train/checkpoint-$steps" \
    --tasks "$data_dir/tasks.jsonl" --split val --out "$output_dir/val_rollouts" \
    --max-steps 8 --max-new-tokens 4096 --system-profile rgb \
    --checkpoint-interface --strict-extent
fi
