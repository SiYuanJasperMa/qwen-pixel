"""Convert GPT-5.5 MagicBrush teacher samples into Qwen Pixel SFT rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
SYSTEM = (
    'You are Qwen Pixel, an image editing agent. Given a source image and an instruction, '
    'produce structured visual reasoning and, when possible, executable Python code. '
    'Use <visual_evidence>, <editing_plan>, and <preservation> sections. '
    'If the edit can be done with PIL/NumPy and the provided helpers, output <code> containing '
    'a function `def edit(image):` that returns a PIL Image. '
    'If the edit requires generating new semantic content, output '
    '<requires_inpaint>true</requires_inpaint> instead of code. '
    'Never use the network, shell, or external files.'
)


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fmt_list(items, prefix='- '):
    return '\n'.join(prefix + str(x) for x in items)


def format_answer(sample, judged, include_code):
    r = sample['reasoning']
    lines = ['<visual_evidence>', fmt_list(r.get('visual_evidence') or []), '</visual_evidence>',
             '<editing_plan>', fmt_list(r.get('editing_plan') or [], ''),
             '</editing_plan>',
             '<preservation>', fmt_list(r.get('preservation_constraints') or []), '</preservation>']
    if sample.get('requires_inpaint'):
        lines.append('<requires_inpaint>true</requires_inpaint>')
    elif include_code and judged.get(sample['id'], {}).get('pass') and sample.get('code'):
        lines += ['<code>', sample['code'].strip(), '</code>']
    elif include_code and sample.get('code'):
        lines.append('<code_status>failed_validation</code_status>')
    return '\n'.join(lines)


def img_id(sample_id):
    parts = sample_id.split('_')
    return parts[2] if len(parts) > 2 else sample_id


def split_for(image_id):
    bucket = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else 'val'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True)
    parser.add_argument('--judged', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['mixed', 'code_pass', 'reasoning'], default='mixed')
    args = parser.parse_args()
    samples = load(args.samples)
    judged = {r['id']: r for r in load(args.judged)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    handles = {split: (out / f'{split}.jsonl').open('w') for split in ['train', 'val']}
    counts = {'train': 0, 'val': 0}
    kept = 0
    for sample in samples:
        j = judged.get(sample['id'], {})
        if args.mode == 'code_pass' and not j.get('pass'):
            continue
        include_code = args.mode in {'mixed', 'code_pass'}
        answer = format_answer(sample, judged, include_code)
        split = split_for(img_id(sample))
        row = {
            'id': sample['id'],
            'task': 'edit_plan' if sample.get('requires_inpaint') else 'edit_code',
            'system': SYSTEM,
            'instruction': 'EDIT: ' + sample['instruction'].strip(),
            'image': str(Path(sample['source_image']).resolve()),
            'answer': answer,
            'meta': {
                'source': 'magicbrush_train_gpt55_teacher',
                'requires_inpaint': bool(sample.get('requires_inpaint')),
                'judge_pass': bool(j.get('pass')),
                'judge_scores': {k: j.get(k) for k in
                                 ['instruction_adherence', 'preservation', 'visual_quality']},
                'execution': sample.get('execution', {}).get('status'),
            },
        }
        handles[split].write(json.dumps(row, ensure_ascii=False) + '\n')
        counts[split] += 1
        kept += 1
    for handle in handles.values():
        handle.close()
    manifest = {'samples': len(samples), 'kept': kept, 'mode': args.mode, 'counts': counts,
                'source': args.samples, 'judged': args.judged}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
st = {'samples': len(samples), 'kept': kept, 'mode': args.mode, 'counts': counts,
                'source': args.samples, 'judged': args.judged}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
