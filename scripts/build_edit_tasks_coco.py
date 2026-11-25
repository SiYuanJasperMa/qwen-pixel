"""Build 10k+ real-photo editing tasks from local COCO/RefCOCO data."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke')
REFCOCO = ROOT / 'data/refcoco_train_real_scale_v1/train.jsonl'
COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'pink']


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def expression(instruction):
    marker = 'object described as:'
    return instruction.split(marker, 1)[1].strip() if marker in instruction else instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/edit_tasks_coco_v1/tasks.jsonl')
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load(REFCOCO)
    tasks = []
    for i, row in enumerate(rows):
        expr = expression(row['instruction'])
        op_index = i % 4
        if op_index == 0:
            color = COLORS[i % len(COLORS)]
            instruction = f'change the object described as "{expr}" to {color}'
            operation = f'recolor_{color}'
        elif op_index == 1:
            instruction = f'remove the object described as "{expr}"'
            operation = 'remove'
        elif op_index == 2:
            instruction = f'blur the object described as "{expr}"'
            operation = 'blur'
        else:
            instruction = f'make the object described as "{expr}" brighter'
            operation = 'brighten'
        tasks.append({
            'id': 'refcoco_edit_' + row['id'],
            'image': row['image'],
            'instruction': instruction,
            'meta': {'source': 'refcoco_train', 'expression': expr, 'operation': operation,
                     'bbox': row.get('bbox'), 'scene_id': row.get('scene_id')},
        })
    images = sorted({row['image'] for row in rows})
    globals_ops = ['convert the image to grayscale', 'flip the image horizontally',
                   'increase the image brightness', 'make the image warmer']
    for i, image in enumerate(images):
        stem = Path(image).stem
        for k in range(2):
            op_index = (i + k) % len(globals_ops)
            instruction = globals_ops[op_index]
            tasks.append({
                'id': f'coco_global_{stem}_{op_index}',
                'image': image,
                'instruction': instruction,
                'meta': {'source': 'coco_train2014', 'operation': instruction},
            })
    out.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks))
    manifest = {'tasks': len(tasks), 'refcoco_edits': len(rows), 'global_edits': len(images),
                'images': len(images), 'operations': dict(Counter(t['meta']['operation'] for t in tasks))}
    (out.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
