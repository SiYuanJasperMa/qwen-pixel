"""Merge one-shot SFT datasets.

- Exact duplicates (same id + same answer) are dropped.
- Different trajectories for the same id inside one input are kept.
- A later input replaces all rows sharing an id from earlier inputs
  (e.g. target-informed rows supersede source-only rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True, help='comma-separated dataset dirs, later wins on id clash')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-replace', action='store_true', help='keep earlier trajectories for the same id')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {'train': {}, 'val': {}}
    for d in [x.strip() for x in args.inputs.split(',') if x.strip()]:
        for split in splits:
            incoming = {}
            for r in load(Path(d) / f'{split}.jsonl'):
                key = (r['id'], hashlib.md5(r.get('answer', '').encode()).hexdigest())
                incoming[key] = r
            ids = {r['id'] for r in incoming.values()}
            if not args.no_replace:
                splits[split] = {k: v for k, v in splits[split].items() if k[0] not in ids}
            splits[split].update(incoming)
    stats = {}
    for split, rows in splits.items():
        data = list(rows.values())
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
        stats[split] = len(data)
    (out / 'manifest.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
