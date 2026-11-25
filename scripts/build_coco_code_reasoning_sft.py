"""Add the teacher's structured reasoning to the COCO edit SFT trajectories.

The code-only rows supervise only the python action. This variant also supervises
the teacher's visual evidence, target, plan and preservation constraints as the
assistant content before the same python tool call.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')


def load_reasoning():
    reasoning = {}
    for pattern in ('data/teacher_coco_*/samples.jsonl', 'data/teacher_gpt55_*/samples.jsonl'):
        for path in glob.glob(str(ROOT / pattern)):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('reasoning'):
                    reasoning[row['id']] = row['reasoning']
    return reasoning


def render(reasoning):
    parts = []
    evidence = reasoning.get('visual_evidence') or []
    if evidence:
        parts.append('<visual_evidence>\n' + '\n'.join(f'- {x}' for x in evidence) + '\n</visual_evidence>')
    target = reasoning.get('target_object')
    region = reasoning.get('target_region')
    intent = reasoning.get('editing_intent')
    if target or region or intent:
        lines = []
        if target:
            lines.append(f'object: {target}')
        if region:
            lines.append(f'region: {region}')
        if intent:
            lines.append(f'intent: {intent}')
        parts.append('<target>\n' + '\n'.join(lines) + '\n</target>')
    plan = reasoning.get('editing_plan') or []
    if plan:
        parts.append('<plan>\n' + '\n'.join(f'{i + 1}. {x}' for i, x in enumerate(plan)) + '\n</plan>')
    preserve = reasoning.get('preservation_constraints') or []
    if preserve:
        parts.append('<preserve>\n' + '\n'.join(f'- {x}' for x in preserve) + '\n</preserve>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', default=str(ROOT / 'data/sft_coco_code_judged_pilot100_v1/rows.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/sft_coco_code_reasoning_v1'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reasoning = load_reasoning()
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    missing = 0
    result = []
    for row in rows:
        tid = row['meta']['trajectory_id']
        text = render(reasoning.get(tid) or {})
        if not text:
            missing += 1
            continue
        row = json.loads(json.dumps(row))
        if row['id'].endswith('_step_00'):
            row['target_message']['content'] = text
        elif row['id'].endswith('_step_01'):
            # The finish row's history must contain the same reasoned assistant action.
            row['messages'][2]['content'] = text
        row['meta']['reasoning_included'] = True
        row['meta']['reasoning_sha256'] = __import__('hashlib').sha256(text.encode()).hexdigest()
        result.append(row)
    train = [r for r in result if r['meta']['split'] == 'train']
    val = [r for r in result if r['meta']['split'] == 'val']
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (out / 'val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in val))
    summary = {'rows': len(result), 'train_rows': len(train), 'val_rows': len(val),
               'trajectories': len(result) // 2, 'missing_reasoning_rows': missing}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
