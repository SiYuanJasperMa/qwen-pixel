"""Balance task counts across one or more JSONL sources."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def load(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT / p
        rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    return rows


def sample(rows, per_task, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['task']].append(row)
    chosen = []
    for task in sorted(grouped):
        pool = grouped[task][:]
        rng.shuffle(pool)
        if len(pool) <= per_task:
            chosen.extend(pool)
        else:
            chosen.extend(pool[:per_task])
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--val', nargs='+', default=[])
    parser.add_argument('--test', nargs='+', default=[])
    parser.add_argument('--per-task-train', type=int, default=3000)
    parser.add_argument('--per-task-val', type=int, default=200)
    parser.add_argument('--per-task-test', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'per_task': {}, 'counts': {}, 'sources': {'train': args.train, 'val': args.val, 'test': args.test}}
    for split, paths, per_task in [('train', args.train, args.per_task_train),
                                   ('val', args.val, args.per_task_val),
                                   ('test', args.test, args.per_task_test)]:
        if not paths:
            continue
        rows = load(paths)
        chosen = sample(rows, per_task, args.seed + {'train': 0, 'val': 1, 'test': 2}[split])
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in chosen))
        counts = defaultdict(int)
        for row in chosen:
            counts[row['task']] += 1
        manifest['counts'][split] = dict(sorted(counts.items()))
        manifest['per_task'][split] = per_task
        print(split, len(chosen), dict(sorted(counts.items())), flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
