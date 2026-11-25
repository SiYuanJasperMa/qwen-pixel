"""RFT-style candidate selection for MagicBrush teacher trajectories.

For every MagicBrush sample we have one base teacher trajectory plus optional
variant trajectories from other teachers. This script scores each candidate by
CLIP similarity to the ground-truth target (plus L1 gain and judge scores),
then keeps the best candidate per sample.

Outputs a judged.jsonl compatible with build_magicbrush_oneshot_sft.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
BASE = ROOT / 'data/teacher_magicbrush_merged_v1/judged.jsonl'
VARIANTS = ROOT / 'data/teacher_magicbrush_variants_v1/variants.jsonl'
CACHE = ROOT / 'data/rft_cache/clip_emb.pt'


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidates():
    rows = []
    for r in load_jsonl(BASE):
        if not (r.get('judge') or {}).get('pass') or not r.get('code'):
            continue
        exec_path = (r.get('execution') or {}).get('output')
        if not exec_path or not Path(exec_path).exists():
            continue
        rows.append({
            'sample_id': r['id'], 'cand_id': r['id'], 'teacher': r.get('teacher_backend', 'base'),
            'instruction': r['instruction'], 'source_image': r['source_image'],
            'target_image': r['target_image'], 'mask_image': r.get('mask_image'),
            'exec_image': exec_path, 'code': r['code'], 'reasoning': r.get('reasoning') or {},
            'judge': r.get('judge') or {}, 'metrics': r.get('metrics') or {},
        })
    for r in load_jsonl(VARIANTS):
        if not (r.get('judge') or {}).get('pass') or not r.get('code'):
            continue
        exec_path = (r.get('execution') or {}).get('output')
        if not exec_path or not Path(exec_path).exists():
            continue
        rows.append({
            'sample_id': r['sample_id'], 'cand_id': r['id'], 'teacher': r.get('teacher', 'variant'),
            'instruction': r['instruction'], 'source_image': r['source_image'],
            'target_image': r['target_image'], 'mask_image': r.get('mask_image'),
            'exec_image': exec_path, 'code': r['code'], 'reasoning': r.get('reasoning') or {},
            'judge': r.get('judge') or {}, 'metrics': r.get('metrics') or {},
        })
    return rows


def load_clip():
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from transformers import CLIPModel, CLIPProcessor
    name = 'openai/clip-vit-base-patch32'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor, device


def embed_paths(model, processor, device, paths, cache, batch=32):
    todo = [p for p in dict.fromkeys(paths) if p not in cache]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        images = [Image.open(p).convert('RGB') for p in chunk]
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.get_image_features(pixel_values=inputs['pixel_values'])
            feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for path, vec in zip(chunk, feats):
            cache[path] = vec.cpu()
        print(f'embedded {min(start + batch, len(todo))}/{len(todo)}', flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'data/sft_magicbrush_rft_v1'))
    parser.add_argument('--clip-weight', type=float, default=1.0)
    parser.add_argument('--l1-weight', type=float, default=10.0)
    parser.add_argument('--judge-weight', type=float, default=0.01)
    parser.add_argument('--min-clip-gain', type=float, default=None,
                        help='optional hard filter on CLIP gain')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = candidates()
    print(f'candidates={len(rows)}', flush=True)
    model, processor, device = load_clip()
    cache = torch.load(CACHE) if CACHE.exists() else {}
    paths = []
    for r in rows:
        paths.extend([r['exec_image'], r['target_image'], r['source_image']])
    embed_paths(model, processor, device, paths, cache)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE)

    def sim(a, b):
        return float(torch.dot(cache[a], cache[b]))

    for r in rows:
        st = sim(r['source_image'], r['target_image'])
        se = sim(r['exec_image'], r['target_image'])
        r['clip_src_target'] = st
        r['clip_exec_target'] = se
        r['clip_gain'] = se - st
        m = r['metrics']
        r['l1_gain'] = m.get('gain', 0.0)
        j = r['judge']
        r['judge_mean'] = sum(float(j.get(k) or 0) for k in
                              ('instruction_adherence', 'preservation', 'visual_quality')) / 15.0
        r['score'] = (args.clip_weight * r['clip_gain']
                      + args.l1_weight * r['l1_gain']
                      + args.judge_weight * r['judge_mean'])

    by_sample = {}
    for r in rows:
        by_sample.setdefault(r['sample_id'], []).append(r)
    selected = []
    for sid, cands in by_sample.items():
        if args.min_clip_gain is not None:
            cands = [c for c in cands if c['clip_gain'] >= args.min_clip_gain]
            if not cands:
                continue
        selected.append(max(cands, key=lambda c: c['score']))

    (out / 'selected.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    base_rows = [c for c in rows if c['cand_id'] == c['sample_id']]
    stats = {
        'samples': len(by_sample), 'selected': len(selected),
        'multi_candidate_samples': sum(1 for c in by_sample.values() if len(c) > 1),
        'selected_by_teacher': {},
        'base_clip_gain_mean': sum(c['clip_gain'] for c in base_rows) / max(1, len(base_rows)),
        'selected_clip_gain_mean': sum(r['clip_gain'] for r in selected) / max(1, len(selected)),
        'selected_clip_gain_positive': sum(1 for r in selected if r['clip_gain'] > 0),
        'selected_l1_gain_positive': sum(1 for r in selected if r['l1_gain'] > 0),
    }
    for r in selected:
        stats['selected_by_teacher'][r['teacher']] = stats['selected_by_teacher'].get(r['teacher'], 0) + 1
    (out / 'report.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
