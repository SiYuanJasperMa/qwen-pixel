"""Filter one-shot SFT rows: drop dev images and sequences that exceed the budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'train'))
from transformers import AutoProcessor  # noqa: E402
from run_pixel_sft import encode  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dev', default=str(ROOT / 'data/magicbrush_dev32_v1/tasks.jsonl'))
    parser.add_argument('--model', default='/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.8-27B')
    parser.add_argument('--max-action', type=int, default=6800)
    args = parser.parse_args()
    data = Path(args.data_dir)
    dev_hashes = {digest(r['image']) for r in
                  (json.loads(line) for line in Path(args.dev).read_text().splitlines() if line.strip())}
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size = {'shortest_edge': 262144, 'longest_edge': 262144}
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (data / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        kept = []
        drop_dev = drop_context = drop_action = 0
        for row in rows:
            if digest(row['image']) in dev_hashes:
                drop_dev += 1
                continue
            batch = encode(processor, row, 'cpu', True, 8192, 1., 1.)
            if batch is None:
                drop_context += 1
                continue
            if int((batch['labels'] != -100).sum()) > args.max_action:
                drop_action += 1
                continue
            kept.append(row)
        (data / f'{split}_filtered.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept))
        print(json.dumps({'split': split, 'rows': len(rows), 'kept': len(kept),
                          'drop_dev': drop_dev, 'drop_context': drop_context,
                          'drop_action': drop_action}), flush=True)


if __name__ == '__main__':
    main()
