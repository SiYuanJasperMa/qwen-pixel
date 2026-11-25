"""Zero-shot ImageNet classification sanity check for Qwen Pixel."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'train'))
sys.path.insert(0, str(PROJECT))

from run_pixel_sft import encode, load  # noqa: E402


def class_name(path: Path) -> str:
    stem = path.stem
    parts = stem.split('_', 1)
    name = parts[1] if len(parts) == 2 else stem
    return name.replace('_', ' ').lower()


def norm(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B')
    parser.add_argument('--data-dir', default='data/imagenet_sample/imagenet-sample-images-master')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', default='outputs/imagenet_zero_shot.jsonl')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT / data_dir
    images = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {'.jpeg', '.jpg', '.png'}])
    if args.limit and len(images) > args.limit:
        step = len(images) / args.limit
        images = [images[int(i * step)] for i in range(args.limit)]
    print(f'images={len(images)}', flush=True)

    model, processor = load(SimpleNamespace(model=args.model, adapter=None, mode='generate'))
    model.eval()
    eos = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
           processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total = 0
    with output.open('w') as handle:
        for i, path in enumerate(images, 1):
            expected = class_name(path)
            row = {
                'system': 'You are a helpful vision assistant.',
                'instruction': 'What is the main object in this image? Answer with one short English noun.',
                'image': str(path),
            }
            inputs = encode(processor, row, model.device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     use_cache=True, eos_token_id=eos,
                                     pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            text = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            answer = norm(text)
            ok = expected in answer or answer in expected
            correct += int(ok)
            total += 1
            result = {'image': path.name, 'expected': expected, 'answer': text.strip(), 'correct': ok}
            handle.write(json.dumps(result) + '\n')
            print(f'{i}/{len(images)} {path.name} expected={expected!r} answer={text.strip()!r} ok={ok}', flush=True)
    summary = {'model': args.model, 'images': total, 'top1': correct / total if total else 0.0,
               'correct': correct, 'output': str(output)}
    (output.parent / 'imagenet_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
