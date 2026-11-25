#!/bin/bash
# Evaluate all teacher models one-shot on MagicBrush dev32 / UltraEdit dev94 / SciForma dev55.
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
LOG=outputs/teacher_eval_logs
mkdir -p "$LOG"

launch() {
  local teacher="$1" workers="$2"
  setsid nohup "$PY" scripts/eval_teachers_on_dev.py --teacher "$teacher" \
    --benches magicbrush,ultraedit,sciforma --workers "$workers" \
    > "$LOG/$teacher.log" 2>&1 < /dev/null &
  echo "launched $teacher workers=$workers"
}

launch gpt61 3
launch gpt56sol 3
launch gpt55 8
launch terra 3
launch luna 3
launch deepseek 8
launch seed21pro 4
launch seed21turbo 4
sleep 2
