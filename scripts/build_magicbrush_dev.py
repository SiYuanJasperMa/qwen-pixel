"""Build an image-disjoint MagicBrush development set for the inpaint SFT smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
PARQUET = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke/data/public_raw/osunlp_MagicBrush/train-00000-of-00051-9fd9f23e2b1cb397.parquet')


def save(data, path, mode='RGB'):
    import io
    image = Image.open(io.BytesIO(data)).convert(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-judged', default=str(ROOT / 'data/teacher_magicbrush_gpt61_inpaint_54_v1/judged.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/magicbrush_dev8_v1'))
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    used = {json.loads(line)['id'] for line in Path(args.train_judged).read_text().splitlines() if line.strip()}
    used_img = {int(i.split('_')[2]) for i in used}
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    table = pq.read_table(PARQUET, columns=['img_id', 'turn_index', 'instruction',
                                            'source_img', 'target_img', 'mask_img'])
    tasks = []
    seen = set()
    for i in range(table.num_rows):
        img_id = int(table['img_id'][i].as_py())
        if img_id in used_img or img_id in seen:
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
                      'split': 'val', 'evaluation_only': True, 'source': 'MagicBrush_train_image_disjoint'})
        seen.add(img_id)
        if len(tasks) >= args.count:
            break
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    print(json.dumps({'tasks': len(tasks), 'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
