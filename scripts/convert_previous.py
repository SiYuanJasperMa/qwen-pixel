"""Convert the completed qwen_image_edit_smoke mixture to Qwen Pixel rows.

Edit rows become palette/grid targets by quantizing their existing target
images. Grounding and QA rows remain structured pixel-understanding tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pixel import prompts  # noqa: E402
from pixel.palette import palette_grid_text, quantize_image  # noqa: E402

EXPERIMENT = PROJECT.parent / 'qwen_image_edit_smoke'


def external(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path('..') / 'qwen_image_edit_smoke' / p)


def split_for(scene: str) -> str:
    bucket = int(hashlib.md5(scene.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else ('val' if bucket < 95 else 'test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(EXPERIMENT / 'data/grounded_edit_real_scale_v2_hsv_correct/mixed_train.jsonl'))
    parser.add_argument('--out', default='data/real_prev_v1')
    parser.add_argument('--grid-size', type=int, default=16, choices=[8, 16, 32, 64])
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()]
    handles = {split: (out / f'{split}.jsonl').open('w') for split in ['train', 'val', 'test']}
    counts = Counter()
    for row in rows:
        scene = row.get('scene_id') or row['id']
        split = split_for(scene)
        image = external(row['image'])
        if row['task'] == 'edit':
            target_path = Path(row['target'])
            if not target_path.is_absolute():
                target_path = EXPERIMENT / target_path
            palette, grid = quantize_image(Image.open(target_path).convert('RGB'), args.grid_size)
            answer = palette_grid_text(palette, grid)
            instruction = row['instruction']
            if instruction.upper().startswith('EDIT:'):
                instruction = f'EDIT {args.grid_size}x{args.grid_size}:' + instruction[5:]
            new = {
                'id': 'prev_' + row['id'],
                'task': 'edit',
                'system': prompts.SYSTEM,
                'instruction': instruction,
                'image': image,
                'answer': answer,
                'meta': {'grid_size': args.grid_size, 'palette_size': len(palette),
                         'source': 'qwen_image_edit_smoke', 'original_id': row['id'],
                         'scene_id': scene, 'target_image': external(row['target']),
                         'operation': row.get('operation', '')},
            }
        elif row['task'] == 'ground':
            new = {
                'id': 'prev_' + row['id'],
                'task': 'ground',
                'system': prompts.SYSTEM,
                'instruction': row['instruction'],
                'image': image,
                'answer': row['answer'],
                'meta': {'grid_size': None, 'source': 'qwen_image_edit_smoke',
                         'original_id': row['id'], 'scene_id': scene,
                         'bbox': row.get('bbox')},
            }
        elif row['task'] == 'qa':
            new = {
                'id': 'prev_' + row['id'],
                'task': 'qa',
                'system': prompts.SYSTEM,
                'instruction': row['instruction'],
                'image': image,
                'answer': row['answer'],
                'meta': {'grid_size': None, 'source': 'qwen_image_edit_smoke',
                         'original_id': row['id'], 'scene_id': scene},
            }
        else:
            continue
        handles[split].write(json.dumps(new) + '\n')
        counts[(split, new['task'])] += 1
    for handle in handles.values():
        handle.close()
    summary = {'source': args.source, 'grid_size': args.grid_size,
               'counts': {f'{split}/{task}': n for (split, task), n in sorted(counts.items())}}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
