#!/bin/bash
# Wait for UltraEdit shards 7/8 parquet downloads, then launch teacher generation.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
D=data/external_edit/UltraEdit_Region_Based_100k/data
mkdir -p outputs
for spec in "7:deepseek:8" "8:gpt55:8"; do
  s=${spec%%:*}; rest=${spec#*:}; t=${rest%%:*}; w=${rest##*:}
  while [ ! -s "$D/RegionBase-0000${s}-of-00093.parquet" ]; do sleep 120; done
  pgrep -f "build_ultraedit_teacher_pilot.py --shard $s " >/dev/null && continue
  setsid nohup $PY scripts/build_ultraedit_teacher_pilot.py --shard "$s" --teacher "$t" \
    --limit 0 --workers "$w" --out "data/ultraedit_teacher_shard${s}_v1" \
    > "outputs/ue_shard${s}.log" 2>&1 < /dev/null &
  echo "launched ultraedit shard=$s teacher=$t" >> outputs/ue_queue.log
done
