#!/bin/bash
# Build v9 = RFT + target-informed (STaR) SFT data, then train on GPU 6 once free.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python

# Wait until the target-informed runs have produced at least 600 trajectories.
while :; do
  total=0
  for d in data/teacher_target_informed_gpt55 data/teacher_target_informed_gpt61_mask data/teacher_target_informed_deepseek; do
    [ -f "$d/judged.jsonl" ] && total=$((total + $(wc -l < "$d/judged.jsonl")))
  done
  [ "$total" -ge 1200 ] && break
  sleep 300
done

"$PY" scripts/build_target_informed_sft.py \
  --dirs data/teacher_target_informed_gpt55,data/teacher_target_informed_gpt61_mask,data/teacher_target_informed_deepseek \
  --out data/sft_target_informed_v1 > outputs/ti_build.log 2>&1

python3 - <<'PY'
import json, random
from pathlib import Path
root = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
out = root / 'data/sft_rft_ti_v1'; out.mkdir(parents=True, exist_ok=True)
for split in ('train', 'val'):
    rows = load(root / f'data/sft_magicbrush_rft_oneshot_v1/{split}.jsonl')
    rows += load(root / f'data/sft_target_informed_v1/{split}.jsonl')
    random.Random(23).shuffle(rows)
    (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(split, len(rows))
PY

while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null || echo 99999)
  [ "$used" -lt 5000 ] && break
  sleep 60
done
mkdir -p outputs/sft_rft_ti_v1/train
setsid nohup env QWEN_PIXEL_TRAIN_GPU=6 bash scripts/train_magicbrush_sft_27b.sh \
  data/sft_rft_ti_v1/train.jsonl outputs/sft_rft_ti_v1/train 1600 \
  > outputs/sft_rft_ti_v1/train.log 2>&1 < /dev/null &
echo "v9 RFT+TI training launched on GPU 6"
