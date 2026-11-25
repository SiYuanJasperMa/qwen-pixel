"""Convert the official MagicBrush dev parquet shards into a task file with images."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
SRC = ROOT / 'data/external_edit/MagicBrush_official/data'


def save(value, path, mode='RGB'):
    data = value.get('bytes') if isinstance(value, dict) else value
    image = Image.open(io.BytesIO(data)).convert(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'data/magicbrush_official_dev_v1'))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    images = out / 'images'
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in sorted(SRC.glob('dev-*-of-00004-*.parquet')):
        table = pq.read_table(shard)
        for i in range(table.num_rows):
            r = {k: table[k][i].as_py() for k in table.column_names}
            sid = f"magicbrush_dev_{r['img_id']}_{r['turn_index']}"
            source = images / f'{sid}_source.png'
            target = images / f'{sid}_target.png'
            mask = images / f'{sid}_mask.png'
            save(r['source_img'], source)
            save(r['target_img'], target)
            save(r['mask_img'], mask, 'L')
            rows.append({'id': sid, 'image': str(source), 'instruction': r['instruction'],
                         'target_image': str(target), 'mask_image': str(mask),
                         'split': 'dev', 'evaluation_only': True, 'dataset': 'magicbrush_official'})
    if args.limit:
        rows = rows[:args.limit]
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    n = len(rows)
    for i in range(4):
        part = rows[i::4]
        (out / f'tasks_{i}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in part))
    print(json.dumps({'tasks': n, 'shards': 4}))


if __name__ == '__main__':
    main()
