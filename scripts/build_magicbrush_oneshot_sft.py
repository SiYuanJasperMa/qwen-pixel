"""Build one-shot MagicBrush SFT rows from accepted teacher trajectories.

Each row is a single assistant answer: the teacher's structured reasoning plus
`executable_code`. No tool calls, no finish action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_multi import SYSTEM  # noqa: E402
from build_teacher_magicbrush_inpaint import PROMPT  # noqa: E402


def split_of(sid):
    return 'val' if int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judged', default=str(ROOT / 'data/teacher_magicbrush_scale_v1/judged.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/sft_magicbrush_oneshot_v1'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.judged).read_text().splitlines() if line.strip()]
    accepted = [r for r in rows if (r.get('judge') or {}).get('pass') and r.get('code')]
    train, val = [], []
    for record in accepted:
        reasoning = record.get('reasoning') or {}
        answer = json.dumps({
            'visual_evidence': reasoning.get('visual_evidence', []),
            'target_object': reasoning.get('target_object', ''),
            'target_region': reasoning.get('target_region', ''),
            'editing_intent': reasoning.get('editing_intent', ''),
            'editing_plan': reasoning.get('editing_plan', []),
            'preservation_constraints': reasoning.get('preservation_constraints', []),
            'executable_code': record['code'],
            'requires_inpaint': record.get('requires_inpaint', True),
        }, ensure_ascii=False)
        row = {
            'id': record['id'],
            'task': 'magicbrush_oneshot',
            'system': SYSTEM,
            'instruction': PROMPT.format(instruction=record['instruction']),
            'image': record['source_image'],
            'answer': answer,
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
