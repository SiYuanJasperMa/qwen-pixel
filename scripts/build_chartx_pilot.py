"""Build a small ChartX scientific-plot editing pilot with exact deterministic edits."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
CHARTX = ROOT / 'data/external_edit/ChartX'
TYPES = ['bar_chart', 'line_chart', 'pie_chart', 'area_chart', 'histogram', 'heatmap']


def code_rotate():
    return "def edit(image):\n    return image.rotate(-90, expand=True)\n"


def code_gray():
    return "def edit(image):\n    return image.convert('L').convert('RGB')\n"


def code_bg():
    return ("def edit(image):\n"
            "    import numpy as np\n"
            "    a=np.array(image.convert('RGB')).copy()\n"
            "    m=(a>245).all(axis=-1)\n"
            "    a[m]=[240,240,240]\n"
            "    return Image.fromarray(a)\n")


def code_border():
    return ("def edit(image):\n"
            "    from PIL import ImageDraw\n"
            "    out=image.convert('RGB').copy()\n"
            "    d=ImageDraw.Draw(out)\n"
            "    w,h=out.size\n"
            "    d.rectangle([0,0,w-1,h-1],outline=(220,30,30),width=4)\n"
            "    return out\n")


EDITS = [
    ('rotate_90', 'Rotate the chart 90 degrees clockwise.', code_rotate, lambda im: im.rotate(-90, expand=True)),
    ('grayscale', 'Convert the chart to grayscale.', code_gray, lambda im: im.convert('L').convert('RGB')),
    ('background_gray', 'Change the chart background to light gray.', code_bg, None),
    ('red_border', 'Add a thin red border around the chart.', code_border, None),
]


def apply_edit(im, name, fn):
    if fn is not None:
        return fn(im)
    if name == 'background_gray':
        a = np.array(im.convert('RGB')).copy()
        m = (a > 245).all(axis=-1)
        a[m] = [240, 240, 240]
        return Image.fromarray(a)
    if name == 'red_border':
        out = im.convert('RGB').copy()
        d = ImageDraw.Draw(out)
        w, h = out.size
        d.rectangle([0, 0, w - 1, h - 1], outline=(220, 30, 30), width=4)
        return out
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-type', type=int, default=10)
    parser.add_argument('--out', default=str(ROOT / 'data/chartx_pilot_v1'))
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    annotations = json.loads((CHARTX / 'ChartX_annotation_val.json').read_text())
    by_type = {}
    for item in annotations:
        by_type.setdefault(item['chart_type'], []).append(item)
    z = zipfile.ZipFile(CHARTX / 'ChartX_png.zip')
    names = set(z.namelist())
    tasks, rows = [], []
    for chart_type in TYPES:
        items = by_type.get(chart_type, [])[:args.per_type]
        for item in items:
            img_path = 'ChartX_png/' + item['img'].lstrip('./')
            if img_path not in names:
                continue
            with z.open(img_path) as handle:
                source = Image.open(io.BytesIO(handle.read())).convert('RGB')
            sid_base = f"chartx_{item['imgname']}"
            source_path = out / 'images' / f'{sid_base}_source.png'
            source.save(source_path)
            for name, instruction, code_fn, target_fn in EDITS:
                sid = f'{sid_base}_{name}'
                target = apply_edit(source, name, target_fn)
                target_path = out / 'images' / f'{sid}_target.png'
                target.save(target_path)
                tasks.append({'id': sid, 'image': str(source_path), 'instruction': instruction,
                              'target_image': str(target_path), 'split': 'val',
                              'chart_type': chart_type, 'evaluation_only': True})
                answer = json.dumps({
                    'visual_evidence': [f'A {chart_type} chart is visible.'],
                    'target_object': 'the chart image',
                    'target_region': 'the full chart',
                    'editing_intent': instruction,
                    'editing_plan': ['Apply the requested chart-level transformation.'],
                    'preservation_constraints': ['Keep the chart content and layout otherwise unchanged.'],
                    'executable_code': code_fn(),
                    'requires_inpaint': False,
                }, ensure_ascii=False)
                rows.append({'id': sid, 'task': 'chart_oneshot',
                             'system': 'You edit chart images by writing Python. Return one JSON object only.',
                             'instruction': 'Edit the attached chart image.\nInstruction: ' + instruction,
                             'image': str(source_path), 'answer': answer,
                             'meta': {'chart_type': chart_type, 'instruction': instruction,
                                      'split': 'val', 'source': 'ChartX_deterministic_edit'}})
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    (out / 'sft.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'tasks': len(tasks), 'sft_rows': len(rows), 'types': TYPES}, indent=2))


if __name__ == '__main__':
    main()
