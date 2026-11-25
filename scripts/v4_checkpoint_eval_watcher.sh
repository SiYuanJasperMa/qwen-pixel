#!/bin/bash
# Continuously evaluate new v4 checkpoints on MagicBrush dev32 (4 parts over 3 GPUs).
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
TRAIN=outputs/sft_magicbrush_27b_v4/train
OUT=outputs/sft_magicbrush_27b_v4
TASKS=data/magicbrush_dev32_v1
GPUS=(2 4 5)

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ]
}

for CKPT in 400 600 800 1000 1200 1400 1600; do
  while [ ! -d "$TRAIN/checkpoint-$CKPT" ]; do
    if ! pgrep -f "run_pixel_sft.py train.*sft_magicbrush_27b_v4" >/dev/null; then
      echo "training exited before checkpoint-$CKPT" >&2
      exit 1
    fi
    sleep 120
  done
  [ -f "$OUT/development_${CKPT}/summary.json" ] && continue
  for gpu in "${GPUS[@]}"; do
    while ! gpu_free "$gpu"; do sleep 60; done
  done
  pids=()
  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" scripts/eval_local_qwen_oneshot.py \
      --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
      --tasks "$TASKS/tasks_$i.jsonl" --out "$OUT/development_${CKPT}_part$i" \
      --limit 8 --max-new-tokens 6144 > "$OUT/development_${CKPT}_part$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  # part 3 runs on the first GPU after part 0 finishes
  CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" scripts/eval_local_qwen_oneshot.py \
    --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
    --tasks "$TASKS/tasks_3.jsonl" --out "$OUT/development_${CKPT}_part3" \
    --limit 8 --max-new-tokens 6144 > "$OUT/development_${CKPT}_part3.log" 2>&1
  "$PY" - "$OUT" "$CKPT" <<'PY'
import json, sys
from pathlib import Path
out, ckpt = Path(sys.argv[1]), sys.argv[2]
parts = []
for i in range(4):
    path = out / f'development_{ckpt}_part{i}' / 'summary.json'
    if path.exists():
        parts.append(json.loads(path.read_text()))
summary = {k: sum(int(p.get(k, 0)) for p in parts) for k in ('tasks', 'parsed', 'executed', 'judge_pass')}
summary['overall_pass_rate'] = summary['judge_pass'] / summary['tasks'] if summary['tasks'] else 0
summary['executed_pass_rate'] = summary['judge_pass'] / summary['executed'] if summary['executed'] else 0
target = out / f'development_{ckpt}'
target.mkdir(parents=True, exist_ok=True)
(target / 'summary.json').write_text(json.dumps(summary, indent=2))
with (target / 'results.jsonl').open('w') as handle:
    for i in range(4):
        path = out / f'development_{ckpt}_part{i}' / 'results.jsonl'
        if path.exists():
            handle.write(path.read_text())
print(ckpt, summary, flush=True)
PY
done
echo "v4 checkpoint watcher done"
