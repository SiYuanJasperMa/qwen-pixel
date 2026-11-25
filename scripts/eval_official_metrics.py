"""Official-style editing metrics for MagicBrush / UltraEdit / SciForma.

Computes L1/L2 (full + mask), PSNR, SSIM, LPIPS, CLIP-I and optional DINO
similarity between the model's edited image and the dataset target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')


def to_tensor(path, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def l1_l2(a, b):
    d = (a - b).abs()
    return float(d.mean()), float((d ** 2).mean())


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(a, b, sigma=1.5, k1=0.01, k2=0.03):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    c1, c2 = (k1 * 1.0) ** 2, (k2 * 1.0) ** 2
    scores = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        mu_x = gaussian_filter(x, sigma)
        mu_y = gaussian_filter(y, sigma)
        sigma_x = gaussian_filter(x * x, sigma) - mu_x ** 2
        sigma_y = gaussian_filter(y * y, sigma) - mu_y ** 2
        sigma_xy = gaussian_filter(x * y, sigma) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        scores.append(float(ssim_map.mean()))
    return float(np.mean(scores))


class ClipScorer:
    def __init__(self, device):
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def embed(self, images):
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            return feats / feats.norm(dim=-1, keepdim=True)

    def sim(self, a, b):
        feats = self.embed([a, b])
        return float((feats[0] * feats[1]).sum())


class LpipsScorer:
    def __init__(self, device):
        import lpips
        self.device = device
        self.model = lpips.LPIPS(net='alex').to(device).eval()

    def __call__(self, a, b):
        with torch.inference_mode():
            return float(self.model(a.unsqueeze(0).to(self.device) * 2 - 1,
                                    b.unsqueeze(0).to(self.device) * 2 - 1))


class DinoScorer:
    def __init__(self, device):
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        from transformers import AutoModel, AutoImageProcessor
        self.device = device
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def sim(self, a, b):
        with torch.inference_mode():
            inputs = self.processor(images=[a, b], return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            feats = out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = {}
    for line in Path(args.tasks).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks[row['id']] = row
    results = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    if args.limit:
        results = results[:args.limit]
    clip = ClipScorer(device)
    lpips_scorer = LpipsScorer(device)
    dino = DinoScorer(device)
    out_dir = Path(args.results).parent
    rows = []
    for r in results:
        t = tasks.get(r['id'])
        if not t or not t.get('target_image'):
            continue
        edited = (r.get('execution') or {}).get('output')
        if not edited or not Path(edited).exists():
            continue
        size = (args.size, args.size)
        src = to_tensor(t['image'], size)
        tgt = to_tensor(t['target_image'], size)
        ed = to_tensor(edited, size)
        l1, l2 = l1_l2(ed, tgt)
        rec = {'id': r['id'], 'l1': l1, 'l2': l2, 'psnr': psnr(ed, tgt),
               'ssim': ssim(ed, tgt), 'lpips': lpips_scorer(ed, tgt),
               'clip_i': clip.sim(edited, t['target_image']),
               'clip_gain': clip.sim(edited, t['target_image']) - clip.sim(t['image'], t['target_image']),
               'dino_sim': dino.sim(edited, t['target_image']),
               'dino_gain': dino.sim(edited, t['target_image']) - dino.sim(t['image'], t['target_image']),
               'no_op_l1': l1_l2(src, tgt)[0]}
        if t.get('mask_image') and Path(t['mask_image']).exists():
            mask = Image.open(t['mask_image']).convert('L').resize(size, Image.Resampling.NEAREST)
            m = torch.from_numpy(np.asarray(mask).astype(np.float32) / 255.0).unsqueeze(0) > 0.5
            if m.any():
                rec['mask_l1'] = float(((ed - tgt).abs() * m).sum() / (m.sum() * 3))
                rec['preserve_l1'] = float(((ed - tgt).abs() * (~m)).sum() / ((~m).sum() * 3))
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    if rows:
        summary = {k: float(np.mean([r[k] for r in rows if k in r]))
                   for k in ('l1', 'l2', 'psnr', 'ssim', 'lpips', 'clip_i', 'clip_gain', 'dino_sim', 'dino_gain',
                             'no_op_l1', 'mask_l1', 'preserve_l1')}
        summary['n'] = len(rows)
    else:
        summary = {'n': 0}
    out = Path(args.out or out_dir / 'official_metrics.json')
    out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
