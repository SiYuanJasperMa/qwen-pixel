"""Execution-grounded reward for Qwen Pixel-Edit RL.

Reward = format + execution + target CLIP gain + target L1 gain + preservation.
No VLM judge in the inner loop; the judge is only used for calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import run_code  # noqa: E402
from build_teacher_multi import metrics  # noqa: E402


def clip_gain(source, target, edited, model, processor, device):
    with torch.inference_mode():
        inputs = processor(images=[Image.open(p).convert('RGB') for p in (source, target, edited)],
                           return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.get_image_features(pixel_values=inputs['pixel_values'])
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[2] * feats[1]).sum() - (feats[0] * feats[1]).sum())


def reward(source, target, mask, code, model, processor, device, out_path):
    r = {'format': 0.0, 'exec': 0.0, 'clip': 0.0, 'l1': 0.0, 'preserve': 0.0}
    if not code or 'def edit(' not in code:
        return r, {'error': 'missing edit()'}
    r['format'] = 0.1
    execution = run_code(code, Path(source), Path(out_path))
    if execution['status'] != 'passed':
        return r, {'error': execution.get('error', 'exec failed')}
    r['exec'] = 0.2
    m = metrics(Path(source), Path(target), Path(execution['output']),
                Path(mask) if mask else None)
    r['clip'] = 0.4 * max(-0.1, min(0.1, clip_gain(source, target, execution['output'],
                                                    model, processor, device)))
    r['l1'] = 0.2 * max(-0.05, min(0.05, m.get('gain', 0.0)))
    preserve = m.get('preserve_l1', 0.0)
    r['preserve'] = 0.1 * max(0.0, 1.0 - min(1.0, preserve / 0.1))
    r['total'] = sum(r[k] for k in ('format', 'exec', 'clip', 'l1', 'preserve'))
    return r, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='results.jsonl from eval_local_qwen_oneshot')
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    import os
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    tasks = {json.loads(l)['id']: json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()][:args.limit]
    out = Path(args.out or Path(args.results).parent / 'rewards.jsonl')
    with out.open('w') as handle:
        for r in rows:
            t = tasks.get(r['id'])
            if not t:
                continue
            code = ''
            raw_path = Path(args.results).parent / f"{r['id']}.txt"
            if raw_path.exists():
                try:
                    code = parse_json(raw_path.read_text()).get('executable_code', '')
                except Exception:
                    code = ''
            if not code:
                continue
            rw, info = reward(t['image'], t['target_image'], t.get('mask_image'), code,
                              model, processor, device,
                              Path(args.results).parent / 'executions' / f"reward_{r['id']}.png")
            record = {'id': r['id'], 'reward': rw, 'metrics': info,
                      'judge_pass': (r.get('judge') or {}).get('pass')}
            handle.write(json.dumps(record) + '\n')
            print(record, flush=True)


if __name__ == '__main__':
    main()
