"""Build image+instruction -> edit(image) code SFT rows from teacher samples."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
SYSTEM = (
    'You are Qwen Pixel, an image editing agent. Given a source image and an instruction, '
    'output only a Python function `def edit(image):` that returns the edited PIL Image. '
    'Use PIL and NumPy. Do not output markdown fences, JSON, or explanations. '
    'If the edit cannot be done with pixel operations, output REQUIRES_INPAINT.'
)
PREFERENCE = {'gpt61': 6, 'gpt55': 5, 'terra': 4, 'luna': 3, 'sol': 2, 'deepseek': 1}


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_for(image):
    bucket = int(hashlib.md5(str(image).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else 'val'


def clean_code(code):
    code = (code or '').strip()
    if code.startswith('```'):
        code = code.split('\n', 1)[1] if '\n' in code else code
        code = code.rsplit('```', 1)[0]
    return code.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/sft_code_coco_v1')
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for path in sorted((ROOT / 'data').glob('teacher_coco_*/samples.jsonl')):
        for row in load(path):
            backend = row.get('teacher_backend', '')
            current = seen.get(row['id'])
            if current is None or PREFERENCE.get(backend, 0) > PREFERENCE.get(current.get('teacher_backend', ''), 0):
                seen[row['id']] = row
    handles = {split: (out / f'{split}.jsonl').open('w') for split in ['train', 'val']}
    counts = {'train': 0, 'val': 0}
    skipped = 0
    for row in seen.values():
        code = clean_code(row.get('code'))
        if row.get('execution', {}).get('status') != 'passed' or not code:
            skipped += 1
            continue
        instruction = row['instruction'].strip()
        if not instruction.upper().startswith('EDIT:'):
            instruction = 'EDIT: ' + instruction
        split = split_for(row['source_image'])
        sft = {
            'id': row['id'],
            'task': 'edit_code',
            'system': SYSTEM,
            'instruction': instruction,
            'image': row['source_image'],
            'answer': code,
            'meta': {'source': 'coco_edit_teacher', 'teacher_backend': row.get('teacher_backend'),
                     'requires_inpaint': row.get('requires_inpaint', False),
                     'execution': row.get('execution', {}).get('status')},
        }
        handles[split].write(json.dumps(sft, ensure_ascii=False) + '\n')
        counts[split] += 1
    for handle in handles.values():
        handle.close()
    manifest = {'unique_teacher_ids': len(seen), 'kept': counts, 'skipped': skipped}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
