"""Score Qwen Pixel generations against saved palette/grid targets."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT))

from pixel.palette import (  # noqa: E402
    grid_exact, palette_exact, parse_palette_grid, pixel_accuracy,
    quantize_image, render_palette_grid,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def norm(text):
    return re.sub(r'\s+', ' ', str(text).strip().strip('"').lower())


def parse_bbox(text):
    match = re.search(r'\{.*?\}', text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if 'bbox' in value and len(value['bbox']) == 4:
        return [float(v) for v in value['bbox']]
    return None


def iou(a, b):
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--gallery-limit', type=int, default=20)
    args = parser.parse_args()

    rows = {r['id']: r for r in read_jsonl(args.data)}
    preds = {r['id']: r for r in read_jsonl(args.predictions)}
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out
    out.mkdir(parents=True, exist_ok=True)
    per_case = []
    grouped = defaultdict(list)
    gallery = []
    for case_id, pred in preds.items():
        row = rows.get(case_id)
        if row is None:
            continue
        task = row['task']
        text = pred['text']
        result = {'id': case_id, 'task': task, 'text': text}
        if task in {'t2p', 'i2p', 'edit'}:
            target_palette, target_grid = parse_palette_grid(row['answer'])
            try:
                pred_palette, pred_grid = parse_palette_grid(text)
                result.update(parsed=True, palette_exact=palette_exact(pred_palette, target_palette),
                              grid_exact=grid_exact(pred_grid, target_grid),
                              pixel_accuracy=pixel_accuracy(pred_grid, target_grid))
                if task == 'edit':
                    source = Image.open(PROJECT / row['image'] if not Path(row['image']).is_absolute() else row['image']).convert('RGB')
                    source_palette, source_grid = quantize_image(source, args.grid_size)
                    changed = preserve = changed_ok = preserve_ok = 0
                    for sr, tr, pr in zip(source_grid, target_grid, pred_grid):
                        for s, t, p in zip(sr, tr, pr):
                            if s == t:
                                preserve += 1
                                preserve_ok += int(p == s)
                            else:
                                changed += 1
                                changed_ok += int(p == t)
                    result.update(
                        changed_accuracy=changed_ok / changed if changed else 1.0,
                        preserve_accuracy=preserve_ok / preserve if preserve else 1.0,
                        no_op_gain=pixel_accuracy(pred_grid, target_grid) - pixel_accuracy(source_grid, target_grid),
                    )
            except Exception as exc:
                result.update(parsed=False, error=str(exc), pixel_accuracy=0.0)
            if len(gallery) < args.gallery_limit:
                try:
                    target_img = render_palette_grid(target_palette, target_grid, scale=16)
                    pred_img = render_palette_grid(pred_palette, pred_grid, scale=16) if result.get('parsed') else Image.new('RGB', (256, 256), 'white')
                    panel = Image.new('RGB', (512, 256), 'white')
                    panel.paste(target_img, (0, 0))
                    panel.paste(pred_img, (256, 0))
                    gallery.append((case_id, panel))
                except Exception:
                    pass
        elif task == 'ground':
            target = row.get('bbox') or (json.loads(row['answer']).get('bbox') if row.get('answer') else None)
            pred_bbox = parse_bbox(text)
            value = iou(pred_bbox, target)
            result.update(parsed=pred_bbox is not None, iou=value, success=value >= 0.5)
        else:
            expected = str(row.get('expected_answer', row.get('answer', '')))
            result.update(parsed=True, exact=norm(text) == norm(expected))
        per_case.append(result)
        grouped[task].append(result)

    summary = {'predictions': len(per_case), 'tasks': {}}
    for task, cases in grouped.items():
        metrics = {'count': len(cases)}
        if task in {'t2p', 'i2p', 'edit'}:
            metrics['parse_rate'] = sum(c.get('parsed', False) for c in cases) / len(cases)
            valid = [c for c in cases if c.get('parsed')]
            if valid:
                metrics['palette_exact_rate'] = sum(c.get('palette_exact', False) for c in valid) / len(valid)
                metrics['grid_exact_rate'] = sum(c.get('grid_exact', False) for c in valid) / len(valid)
                metrics['pixel_accuracy'] = sum(c.get('pixel_accuracy', 0.0) for c in valid) / len(valid)
                if task == 'edit':
                    metrics['changed_accuracy'] = sum(c.get('changed_accuracy', 0.0) for c in valid) / len(valid)
                    metrics['preserve_accuracy'] = sum(c.get('preserve_accuracy', 0.0) for c in valid) / len(valid)
                    metrics['no_op_gain'] = sum(c.get('no_op_gain', 0.0) for c in valid) / len(valid)
        elif task == 'ground':
            metrics['success_rate'] = sum(c.get('success', False) for c in cases) / len(cases)
            metrics['mean_iou'] = sum(c.get('iou', 0.0) for c in cases) / len(cases)
        else:
            metrics['exact_rate'] = sum(c.get('exact', False) for c in cases) / len(cases)
        summary['tasks'][task] = metrics

    (out / 'per_case.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in per_case))
    (out / 'eval_summary.json').write_text(json.dumps(summary, indent=2))
    if gallery:
        sheet = Image.new('RGB', (512, 256 * len(gallery)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (case_id, panel) in enumerate(gallery):
            sheet.paste(panel, (0, i * 256))
            draw.text((4, i * 256 + 4), case_id[:90], fill='black')
        sheet.save(out / 'gallery.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
