"""Build COCO pixel-understanding tasks from local RefCOCO annotations."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
REFCOCO = ROOT / 'qwen_image_edit_smoke/data/refcoco_train_real_scale_v1/train.jsonl'
OUT = ROOT / 'qwen_pixel/data/understanding_coco_v1'
SYSTEM = (
    'You are Qwen Pixel, a visual understanding model. Answer questions about the image '
    'directly and briefly. For GROUND tasks output only JSON: {"bbox": [x1,y1,x2,y2]}. '
    'Do not use markdown fences.'
)


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_bbox(value):
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    return [float(v) for v in value]


def split_for(scene_id):
    bucket = int(hashlib.md5(str(scene_id).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else ('val' if bucket < 95 else 'test')


def expression(instruction):
    marker = 'object described as:'
    return instruction.split(marker, 1)[1].strip() if marker in instruction else instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    rows = load(REFCOCO)
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[str(row.get('scene_id') or row['id'])].append(row)
    scenes = sorted(by_scene)
    rng = random.Random(42)
    handles = {split: (out / f'{split}.jsonl').open('w') for split in ['train', 'val', 'test']}
    counts = defaultdict(int)
    all_expressions = [(str(r.get('scene_id') or r['id']), expression(r['instruction'])) for r in rows]
    for scene_id in scenes:
        scene_rows = by_scene[scene_id]
        image_path = Path(scene_rows[0]['image'])
        if not image_path.is_absolute():
            image_path = ROOT / 'qwen_image_edit_smoke' / image_path
        dest = out / 'images' / f'{scene_id}.png'
        if not dest.exists():
            Image.open(image_path).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS).save(dest)
        split = split_for(scene_id)
        handle = handles[split]
        boxes = {}
        for row in scene_rows:
            bbox = parse_bbox(row['bbox'])
            boxes[tuple(round(v, 2) for v in bbox)] = bbox
        for row in scene_rows:
            expr = expression(row['instruction'])
            bbox = [round(v * 2, 1) for v in parse_bbox(row['bbox'])]
            task = {
                'id': f'understand_ground_{row["id"]}',
                'task': 'ground', 'system': SYSTEM,
                'instruction': f'GROUND: Locate the object described as: {expr}',
                'image': str(dest), 'answer': json.dumps({'bbox': bbox}, ensure_ascii=False),
                'meta': {'source': 'refcoco_train', 'scene_id': scene_id, 'bbox': bbox},
            }
            handle.write(json.dumps(task, ensure_ascii=False) + '\n')
            counts[('ground', split)] += 1
        # Count of distinct annotated regions.
        count = len(boxes)
        task = {
            'id': f'understand_count_{scene_id}', 'task': 'qa', 'system': SYSTEM,
            'instruction': 'QA: How many distinct annotated object regions are in this image?',
            'image': str(dest), 'answer': str(count),
            'meta': {'source': 'refcoco_train', 'scene_id': scene_id, 'operation': 'count'},
        }
        handle.write(json.dumps(task, ensure_ascii=False) + '\n')
        counts[('qa_count', split)] += 1
        # Left/right position for up to two expressions.
        for i, row in enumerate(scene_rows[:2]):
            expr = expression(row['instruction'])
            bbox = parse_bbox(row['bbox'])
            cx = (bbox[0] + bbox[2]) / 2
            side = 'left' if cx < 128 else 'right'
            task = {
                'id': f'understand_position_{row["id"]}', 'task': 'qa', 'system': SYSTEM,
                'instruction': f'QA: Is the object described as "{expr}" on the left or right half of the image?',
                'image': str(dest), 'answer': side,
                'meta': {'source': 'refcoco_train', 'scene_id': scene_id, 'operation': 'position'},
            }
            handle.write(json.dumps(task, ensure_ascii=False) + '\n')
            counts[('qa_position', split)] += 1
        # Presence: one true and one false expression.
        true_expr = expression(scene_rows[0]['instruction'])
        other_scene, other_expr = rng.choice(all_expressions)
        if other_scene == scene_id:
            other_expr = 'a purple elephant'
        for expr, answer in [(true_expr, 'yes'), (other_expr, 'no')]:
            task = {
                'id': f'understand_presence_{scene_id}_{answer}', 'task': 'qa', 'system': SYSTEM,
                'instruction': f'QA: Is there an object described as "{expr}" in this image? Answer yes or no.',
                'image': str(dest), 'answer': answer,
                'meta': {'source': 'refcoco_train', 'scene_id': scene_id, 'operation': 'presence'},
            }
            handle.write(json.dumps(task, ensure_ascii=False) + '\n')
            counts[(f'qa_presence_{answer}', split)] += 1
    for handle in handles.values():
        handle.close()
    manifest = {'scenes': len(scenes), 'counts': {f'{k[0]}/{k[1]}': v for k, v in sorted(counts.items())}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
