"""Build mixed v3 SFT data: MagicBrush v4 + UltraEdit + SciForma, shuffled."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')

SOURCES = {
    'magicbrush': ('data/sft_magicbrush_oneshot_v4/train_filtered.jsonl',
                   'data/sft_magicbrush_oneshot_v4/val_filtered.jsonl'),
    'ultraedit': ('data/sft_ultraedit_oneshot_full_v1/train.jsonl',
                  'data/sft_ultraedit_oneshot_full_v1/val.jsonl'),
    'sciforma': ('data/sft_sciforma_oneshot_full_v1/train.jsonl',
                 'data/sft_sciforma_oneshot_full_v1/val.jsonl'),
}


def load(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'data/sft_mixed_v3'))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--ultraedit-cap', type=int, default=0)
    parser.add_argument('--sciforma-cap', type=int, default=0)
    args = parser.parse_args()
    caps = {'ultraedit': args.ultraedit_cap, 'sciforma': args.sciforma_cap}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ('train', 'val'):
        rows, seen = [], set()
        for name, paths in SOURCES.items():
            part = load(paths[0] if split == 'train' else paths[1])
            cap = caps.get(name, 0)
            if cap:
                part = random.Random(args.seed).sample(part, min(cap, len(part)))
            counts[f'{name}_{split}'] = len(part)
            for row in part:
                if row['id'] in seen:
                    continue
                seen.add(row['id'])
                rows.append(row)
        random.Random(args.seed).shuffle(rows)
        (out / f'{split}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        counts[f'total_{split}'] = len(rows)
    (out / 'counts.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
