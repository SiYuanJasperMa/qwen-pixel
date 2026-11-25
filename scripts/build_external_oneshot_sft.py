"""Convert accepted UltraEdit/SciForma teacher records to one-shot SFT rows and dev tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_multi import SYSTEM
from build_teacher_magicbrush_inpaint import PROMPT
from pixel.data_lineage import source_group, split_of_group


def split_of(sid, dev_count):
    return 'val' if int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 100 < dev_count else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judged', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dev-percent', type=int, default=10)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.judged).read_text().splitlines() if line.strip()]
    accepted = [r for r in rows if (r.get('judge') or {}).get('pass') and r.get('code')]
    train, dev = [], []
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
            'requires_inpaint': record.get('requires_inpaint', False),
        }, ensure_ascii=False)
        group = source_group(record['id'], record['source_image'])
        split = split_of_group(group)
        row = {'id': record['id'], 'task': f'{args.dataset}_oneshot', 'system': SYSTEM,
               'instruction': PROMPT.format(instruction=record['instruction']),
               'image': record['source_image'], 'answer': answer,
               'meta': {'instruction': record['instruction'], 'split': split,
                        'dataset': args.dataset, 'source': f'{args.dataset}_teacher',
                        'source_group_id': group, 'generation_mode': 'rft',
                        'teacher_backend': record.get('teacher_backend', 'unknown')}}
        (dev if split == 'val' else train).append(row)
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (out / 'val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in dev))
    tasks = [{'id': r['id'], 'image': r['image'], 'instruction': r['meta']['instruction'],
              'target_image': next(x['target_image'] for x in accepted if x['id'] == r['id']),
              'split': 'val', 'evaluation_only': True, 'dataset': args.dataset}
             for r in dev]
    (out / 'dev_tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    print(json.dumps({'dataset': args.dataset, 'accepted': len(accepted),
                      'train': len(train), 'dev': len(dev)}, indent=2))


if __name__ == '__main__':
    main()
