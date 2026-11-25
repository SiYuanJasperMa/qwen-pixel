#!/bin/bash
# Cross-benchmark evaluation: best MagicBrush adapter and zero-shot base on
# UltraEdit dev94 and SciForma dev55. Each job pinned to one GPU.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
BEST=outputs/sft_magicbrush_27b_oneshot_v3_cont/train/checkpoint-1200
mkdir -p outputs/crossbench_v3cont1200_ultraedit outputs/crossbench_v3cont1200_sciforma \
         outputs/crossbench_zero_ultraedit outputs/crossbench_zero_sciforma

run() {
  local gpu="$1" tag="$2" tasks="$3" limit="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --tasks "$tasks" --out "outputs/$tag" --limit "$limit" \
    --max-new-tokens 6144 "$@" > "outputs/$tag.log" 2>&1
  echo "done $tag"
}

# GPU 2 already running v3-cont1200 on UltraEdit (launched separately); skip it.
run 3 crossbench_v3cont1200_sciforma data/sft_sciforma_oneshot_full_v1/dev_tasks.jsonl 55 --adapter "$BEST" &
run 4 crossbench_zero_ultraedit data/sft_ultraedit_oneshot_full_v1/dev_tasks.jsonl 94 &
run 5 crossbench_zero_sciforma data/sft_sciforma_oneshot_full_v1/dev_tasks.jsonl 55 &
wait
echo "crossbench evals done"
