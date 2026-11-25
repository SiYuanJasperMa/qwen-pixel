"""Build one-shot SFT rows from target-informed (STaR) teacher records.

Keeps judge-passed trajectories whose code is not hardcoded and which pass the
flip-based reusability probe. The student prompt stays source-only, exactly as at
inference; the target was privileged information used only during generation.
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
from build_magicbrush_target_informed import looks_hardcoded  # noqa: E402
from pixel.data_lineage import canonical_task, source_group, split_of_group


def split_of(sid):
    return 'val' if int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', required=True)
    parser.add_argument('--out', default=str(ROOT / 'data/sft_target_informed_v1'))
    parser.add_argument('--allow-no-reuse', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    stats = {'loaded': 0, 'judge_pass': 0, 'hardcoded': 0, 'reuse_fail': 0, 'kept': 0}
    for d in [x.strip() for x in args.dirs.split(',') if x.strip()]:
        path = Path(d) / 'judged.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            stats['loaded'] += 1
            if not (r.get('judge') or {}).get('pass') or not r.get('code'):
                continue
            stats['judge_pass'] += 1
            hardcoded, _ = looks_hardcoded(r['code'])
            if hardcoded:
                stats['hardcoded'] += 1
                continue
            if not args.allow_no_reuse and r.get('reuse_exec') not in ('passed',):
                stats['reuse_fail'] += 1
                continue
            sid = r['id']
            if sid in seen:
                continue
            seen.add(sid)
            task_id = r.get('task_id', canonical_task(sid))
            group = r.get('source_group_id') or source_group(task_id, r['source_image'])
            split = split_of_group(group)
            mode = r.get('generation_mode', 'ti')
            reasoning = r.get('reasoning') or {}
            answer = json.dumps({
                'visual_evidence': reasoning.get('visual_evidence', []),
                'target_object': reasoning.get('target_object', ''),
                'target_region': reasoning.get('target_region', ''),
                'editing_intent': reasoning.get('editing_intent', ''),
                'editing_plan': reasoning.get('editing_plan', []),
                'preservation_constraints': reasoning.get('preservation_constraints', []),
                'executable_code': r['code'],
                'requires_inpaint': r.get('requires_inpaint', True),
            }, ensure_ascii=False)
            rows.append({
                'id': sid, 'task': r.get('dataset', 'magicbrush') + '_oneshot', 'system': SYSTEM,
                'instruction': PROMPT.format(instruction=r['instruction']),
                'image': r['source_image'], 'answer': answer,
                'meta': {'instruction': r['instruction'], 'teacher_backend': r.get('teacher'),
                         'target_informed': mode == 'ti', 'split': split,
                         'task_id': task_id, 'source_group_id': group, 'generation_mode': mode,
                         'reuse_exec': r.get('reuse_exec')},
            })
            stats['kept'] += 1
    train = [r for r in rows if r['meta']['split'] == 'train']
    val = [r for r in rows if r['meta']['split'] == 'val']
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (out / 'val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in val))
    stats.update({'train_rows': len(train), 'val_rows': len(val)})
    (out / 'manifest.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
