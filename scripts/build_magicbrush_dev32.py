"""Build a 32-image image-disjoint MagicBrush development set."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
SHARDS = '/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke/data/magicbrush_hf/data/train-*.parquet'


def save(data, path, mode='RGB'):
    import io
    image = Image.open(io.BytesIO(data)).convert(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'data/magicbrush_dev32_v1'))
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    used = set()
    existing = ROOT / 'data/magicbrush_dev8_v1/tasks.jsonl'
    tasks = []
    if existing.exists():
        for line in existing.read_text().splitlines():
            if line.strip():
                task = json.loads(line)
                tasks.append(task)
                used.add(int(task['id'].split('_')[2]))
    for shard in sorted(glob.glob(SHARDS)):
        if len(tasks) >= args.count:
            break
        table = pq.read_table(shard, columns=['img_id', 'turn_index', 'instruction',
                                              'source_img', 'target_img', 'mask_img'])
        for i in range(table.num_rows):
            if len(tasks) >= args.count:
                break
            img_id = int(table['img_id'][i].as_py())
            if img_id in used:
                continue
            turn = table['turn_index'][i].as_py()
            sid = f'magicbrush_dev_{img_id}_{turn}'
            source = out / 'images' / f'{sid}_source.png'
            target = out / 'images' / f'{sid}_target.png'
            mask = out / 'images' / f'{sid}_mask.png'
            save(table['source_img'][i].as_py()['bytes'], source, 'RGB')
            save(table['target_img'][i].as_py()['bytes'], target, 'RGB')
            save(table['mask_img'][i].as_py()['bytes'], mask, 'L')
            tasks.append({'id': sid, 'image': str(source), 'instruction': table['instruction'][i].as_py(),
                          'target_image': str(target), 'mask_image': str(mask),
                          'split': 'val', 'evaluation_only': True,
                          'source': 'MagicBrush_train_image_disjoint'})
            used.add(img_id)
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    print(json.dumps({'tasks': len(tasks), 'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
