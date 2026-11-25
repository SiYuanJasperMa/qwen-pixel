"""Build code-only one-shot MagicBrush SFT rows (no JSON reasoning wrapper)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')

SYSTEM = 'You are Qwen Pixel. Write Python code that edits the attached RGB image. Return only the code.'
INSTRUCTION = '''Instruction: {instruction}

Write a complete Python function `def edit(image):` that returns a PIL Image of the same size.
The source image is 512x512. Use only PIL, numpy, cv2, math, json, or colorsys.
Do not use external models, network access, or external files.
Return only the code, without markdown fences.'''


def split_of(sid):
    return 'val' if int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judged', default=str(ROOT / 'data/teacher_magicbrush_scale_v1/judged.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/sft_magicbrush_oneshot_code_v1'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.judged).read_text().splitlines() if line.strip()]
    accepted = [r for r in rows if (r.get('judge') or {}).get('pass') and r.get('code')]
    train, val = [], []
    for record in accepted:
        row = {
            'id': record['id'],
            'task': 'magicbrush_oneshot_code',
            'system': SYSTEM,
            'instruction': INSTRUCTION.format(instruction=record['instruction']),
            'image': record['source_image'],
            'answer': record['code'],
            'meta': {
                'instruction': record['instruction'],
                'teacher_backend': record.get('teacher_backend'),
                'judge_scores': {k: (record.get('judge') or {}).get(k) for k in
                                 ('instruction_adherence', 'preservation', 'visual_quality', 'pass')},
                'split': split_of(record['id']),
            },
        }
        (train if row['meta']['split'] == 'train' else val).append(row)
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (out / 'val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in val))
    summary = {'accepted': len(accepted), 'train_rows': len(train), 'val_rows': len(val)}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
