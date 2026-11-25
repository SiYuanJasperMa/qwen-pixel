#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
export CUDA_VISIBLE_DEVICES=${1:?Specify available physical GPU}
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
output_dir=outputs/agent_sft_heldout_cache_parity_20261125
mkdir -p "$output_dir"
for case_id in clean_development_00 clean_development_02; do
  if [[ -e "$output_dir/$case_id.json" ]]; then
    echo 'Inspect existing numerical audit before restart.' >&2
    exit 2
  fi
  "$python_bin" scripts/audit_cached_teacher_logits.py \
    --model /mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B \
    --adapter outputs/agent_sft_native_feedback_feedback_20261125/train/checkpoint-450 \
    --data data/agent_sft_heldout_cache_audit_20261125/prefixes.jsonl \
    --id "${case_id}_generated_python" --tokens 4096 \
    --out "$output_dir/$case_id.json"
done
