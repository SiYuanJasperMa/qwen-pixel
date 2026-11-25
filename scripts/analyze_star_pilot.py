"""Compare STaR target-informed teacher data against source-only base data.

Both arms are matched by sample id. Reports execution rate, judge pass rate,
hardcoded-code rate, flip-probe pass rate, L1 gain and CLIP gain vs target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_multi import metrics  # noqa: E402


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def clip_features(paths):
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from transformers import CLIPModel, CLIPProcessor
    name = 'openai/clip-vit-base-patch32'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    feats = {}
    unique = [p for p in dict.fromkeys(paths) if p and Path(p).exists()]
    for start in range(0, len(unique), 32):
        chunk = unique[start:start + 32]
        images = [Image.open(p).convert('RGB') for p in chunk]
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.get_image_features(pixel_values=inputs['pixel_values'])
            vecs = out if isinstance(out, torch.Tensor) else out.pooler_output
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        for path, vec in zip(chunk, vecs):
            feats[path] = vec.cpu()
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--star-dir', required=True)
    parser.add_argument('--base', default=str(ROOT / 'data/teacher_magicbrush_merged_v1/judged.jsonl'))
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    star = load(Path(args.star_dir) / 'judged.jsonl')
    base = {r['id']: r for r in load(args.base)}
    rows = []
    for r in star:
        b = base.get(r['id'])
        if not b:
            continue
        out = (r.get('execution') or {}).get('output')
        rec = {'id': r['id'], 'teacher': r['teacher'],
               'star_status': r['status'],
               'star_pass': bool((r.get('judge') or {}).get('pass')),
               'hardcoded': bool(r.get('hardcoded')),
               'reuse_exec': r.get('reuse_exec'),
               'base_pass': bool((b.get('judge') or {}).get('pass'))}
        if out and Path(out).exists():
            m = metrics(Path(r['source_image']), Path(r['target_image']), Path(out),
                        Path(r['mask_image']) if r.get('mask_image') else None)
            rec.update({f'star_{k}': v for k, v in m.items()})
        bm = b.get('metrics') or {}
        rec.update({f'base_{k}': v for k, v in bm.items()})
        rows.append(rec)
    feats = clip_features(
        [r['source_image'] for r in star] + [r['target_image'] for r in star] +
        [(r.get('execution') or {}).get('output') for r in star] +
        [(base[r['id']].get('execution') or {}).get('output') for r in star if r['id'] in base])
    def sim(a, b):
        return float(torch.dot(feats[a], feats[b])) if a in feats and b in feats else None
    for r, s in zip(rows, star):
        out = (s.get('execution') or {}).get('output')
        b = base.get(s['id']) or {}
        b_out = (b.get('execution') or {}).get('output')
        base_sim = sim(s['source_image'], s['target_image'])
        r['star_clip_gain'] = (sim(out, s['target_image']) - base_sim) if out and out in feats else None
        r['base_clip_gain'] = (sim(b_out, s['target_image']) - base_sim) if b_out and b_out in feats else None
    summary = {
        'n': len(rows),
        'star_pass': sum(1 for r in rows if r['star_pass']),
        'base_pass': sum(1 for r in rows if r['base_pass']),
        'hardcoded': sum(1 for r in rows if r['hardcoded']),
        'reuse_passed': sum(1 for r in rows if r['reuse_exec'] == 'passed'),
        'star_l1_gain_mean': sum(r.get('star_gain', 0) for r in rows) / max(1, len(rows)),
        'base_l1_gain_mean': sum(r.get('base_gain', 0) for r in rows) / max(1, len(rows)),
        'star_clip_gain_mean': sum(r.get('star_clip_gain') or 0 for r in rows) / max(1, len(rows)),
        'base_clip_gain_mean': sum(r.get('base_clip_gain') or 0 for r in rows) / max(1, len(rows)),
    }
    out = Path(args.out or Path(args.star_dir) / 'comparison.json')
    out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
