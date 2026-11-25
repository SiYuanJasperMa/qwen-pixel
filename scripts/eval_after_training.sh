#!/bin/bash
# Evaluate checkpoints of one run sequentially on a single GPU, after training.
# usage: eval_after_training.sh <train_dir> <out_dir> <gpu> <ckpt1,ckpt2,...>
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
MODEL=/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B
TRAIN="$1"; OUT="$2"; GPU="$3"; CKPTS="$4"
TASKS=data/magicbrush_dev32_v1

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ]
}

for CKPT in $(echo "$CKPTS" | tr ',' ' '); do
  while [ ! -d "$TRAIN/checkpoint-$CKPT" ]; do sleep 120; done
  [ -f "$OUT/development_${CKPT}/summary.json" ] && continue
  while ! gpu_free; do sleep 60; done
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/eval_local_qwen_oneshot.py \
      --model "$MODEL" --adapter "$TRAIN/checkpoint-$CKPT" \
      --tasks "$TASKS/tasks_$i.jsonl" --out "$OUT/development_${CKPT}_part$i" \
      --limit 8 --max-new-tokens 4096 > "$OUT/development_${CKPT}_part$i.log" 2>&1
  done
  "$PY" - "$OUT" "$CKPT" <<'PY'
import json, sys
from pathlib import Path
out, ckpt = Path(sys.argv[1]), sys.argv[2]
parts = []
for i in range(2):
    path = out / f'development_{ckpt}_part{i}' / 'summary.json'
    if path.exists():
        parts.append(json.loads(path.read_text()))
summary = {k: sum(int(p.get(k, 0)) for p in parts) for k in ('tasks', 'parsed', 'executed', 'judge_pass')}
summary['overall_pass_rate'] = summary['judge_pass'] / summary['tasks'] if summary['tasks'] else 0
summary['executed_pass_rate'] = summary['judge_pass'] / summary['executed'] if summary['executed'] else 0
target = out / f'development_{ckpt}'
target.mkdir(parents=True, exist_ok=True)
(target / 'summary.json').write_text(json.dumps(summary, indent=2))
print(ckpt, summary, flush=True)
PY
done
echo "eval_after_training done for $OUT"
