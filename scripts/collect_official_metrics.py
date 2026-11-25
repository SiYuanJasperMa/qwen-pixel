"""Aggregate official metrics for every completed eval part under outputs/.

Loads CLIP/LPIPS/DINO once, then scores every
outputs/<run>/development_<ckpt>_part<i>/results.jsonl against the dev tasks.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_official_metrics import (ClipScorer, DinoScorer, LpipsScorer,  # noqa: E402
                                   l1_l2, psnr, ssim, to_tensor)

TASKS = {
    'part0': ROOT / 'data/magicbrush_dev32_v1/tasks_0.jsonl',
    'part1': ROOT / 'data/magicbrush_dev32_v1/tasks_1.jsonl',
}
KEYS = ('l1', 'l2', 'psnr', 'ssim', 'lpips', 'clip_i', 'clip_gain', 'dino_sim', 'dino_gain',
        'no_op_l1', 'mask_l1', 'preserve_l1')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip = ClipScorer(device)
    lpips_scorer = LpipsScorer(device)
    dino = DinoScorer(device)
    tasks = {}
    for part, path in TASKS.items():
        tasks[part] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[part][row['id']] = row
    table = {}
    for path in sorted(glob.glob(str(ROOT / 'outputs/*/development_*_part*/results.jsonl'))):
        p = Path(path)
        part = p.parent.name.split('_')[-1]
        if part not in tasks:
            continue
        run = p.parent.parent.name
        ck = p.parent.name.replace('development_', '').replace(f'_{part}', '')
        key = f'{run}/ckpt-{ck}'
        entry = table.setdefault(key, {'n': 0, **{k: 0.0 for k in KEYS}})
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t = tasks[part].get(r['id'])
            if not t or not t.get('target_image'):
                continue
            edited = (r.get('execution') or {}).get('output')
            if not edited or not Path(edited).exists():
                continue
            size = (512, 512)
            src = to_tensor(t['image'], size)
            tgt = to_tensor(t['target_image'], size)
            ed = to_tensor(edited, size)
            l1, l2 = l1_l2(ed, tgt)
            rec = {'l1': l1, 'l2': l2, 'psnr': psnr(ed, tgt), 'ssim': ssim(ed, tgt),
                   'lpips': lpips_scorer(ed, tgt),
                   'clip_i': clip.sim(edited, t['target_image']),
                   'clip_gain': clip.sim(edited, t['target_image']) - clip.sim(t['image'], t['target_image']),
                   'dino_sim': dino.sim(edited, t['target_image']),
                   'dino_gain': dino.sim(edited, t['target_image']) - dino.sim(t['image'], t['target_image']),
                   'no_op_l1': l1_l2(src, tgt)[0]}
            if t.get('mask_image') and Path(t['mask_image']).exists():
                mask = Image.open(t['mask_image']).convert('L').resize(size, Image.Resampling.NEAREST)
                m = torch.from_numpy(np.asarray(mask).astype(np.float32) / 255.0).unsqueeze(0) > 0.5
                if m.any():
                    rec['mask_l1'] = float(((ed - tgt).abs() * m).sum() / (m.sum() * 3))
                    rec['preserve_l1'] = float(((ed - tgt).abs() * (~m)).sum() / ((~m).sum() * 3))
            for k in KEYS:
                entry[k] += rec.get(k, 0.0)
            entry['n'] += 1
    for entry in table.values():
        n = max(1, entry['n'])
        for k in KEYS:
            entry[k] = round(entry[k] / n, 4)
    (ROOT / 'outputs/official_metrics_table.json').write_text(json.dumps(table, indent=2, sort_keys=True))
    for k in sorted(table, key=lambda x: -table[x]['clip_gain']):
        e = table[k]
        print(f"{k:48s} n={e['n']:2d} l1={e['l1']:.4f} psnr={e['psnr']:.2f} "
              f"clip_gain={e['clip_gain']:+.4f} dino_gain={e['dino_gain']:+.4f} noop={e['no_op_l1']:.4f}")


if __name__ == '__main__':
    main()
