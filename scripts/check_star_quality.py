"""Compare target fidelity of accepted target-informed (STaR) trajectories vs the
source-only base trajectory for the same MagicBrush sample."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import run_code  # noqa: E402
from build_teacher_multi import metrics  # noqa: E402

BASE = {json.loads(l)['id']: json.loads(l)
        for l in (ROOT / 'data/teacher_magicbrush_merged_v1/judged.jsonl').read_text().splitlines()
        if l.strip()}


def main():
    teachers = ['deepseek', 'gpt55', 'gpt61_mask']
    sample_n = 50
    tmp = Path('/tmp/ti_quality'); tmp.mkdir(exist_ok=True)
    for t in teachers:
        path = ROOT / f'data/teacher_target_informed_{t}/judged.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if (r.get('judge') or {}).get('pass') and r.get('code') and not r.get('hardcoded')]
        random.Random(0).shuffle(rows)
        rows = rows[:sample_n]
        gains, base_gains, l1s, base_l1s = [], [], [], []
        for r in rows:
            src = Path(r['source_image']); tgt = Path(r['target_image'])
            mask = Path(r['mask_image']) if r.get('mask_image') else None
            out = tmp / f"{r['id']}_{t}.png"
            ex = run_code(r['code'], src, out)
            if ex['status'] != 'passed':
                continue
            m = metrics(src, tgt, Path(ex['output']), mask)
            b = BASE.get(r['id'], {}).get('metrics') or {}
            gains.append(m.get('gain', 0.0)); l1s.append(m.get('l1', 0.0))
            if b:
                base_gains.append(b.get('gain', 0.0)); base_l1s.append(b.get('l1', 0.0))
            try:
                Path(ex['output']).unlink()
            except Exception:
                pass
        n = len(gains)
        print(json.dumps({
            'teacher': t, 'n': n,
            'ti_l1_mean': sum(l1s) / n if n else None,
            'ti_gain_mean': sum(gains) / n if n else None,
            'ti_gain_positive': sum(1 for g in gains if g > 0) / n if n else None,
            'base_l1_mean': sum(base_l1s) / len(base_l1s) if base_l1s else None,
            'base_gain_mean': sum(base_gains) / len(base_gains) if base_gains else None,
        }, indent=2), flush=True)


if __name__ == '__main__':
    main()
