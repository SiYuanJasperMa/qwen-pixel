"""Re-judge accepted COCO edit trajectories with the strict full-target-extent rubric.

The first conversion used the legacy judge, which can accept a partial mask.
This pass keeps only trajectories whose edit covers the full requested object.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT))
from pixel.teacher_api import judge_edit, make_client  # noqa: E402


def judge_one(record, client):
    tid = record['meta']['trajectory_id']
    ws = Path(record['messages'][1]['content'][0]['image']).parent
    source = ws / 'source.png'
    edited = ws / 'edited.png'
    if not source.exists() or not edited.exists():
        return {'id': tid, 'pass': False, 'reason': 'missing source/edited'}
    judgment = judge_edit(client, str(source), str(edited), record['meta']['instruction'],
                          strict_extent=True)
    return {'id': tid, 'pass': bool(judgment.get('pass')), 'judge': judgment}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', default=str(ROOT / 'data/sft_coco_code_judged_pilot100_v1/rows.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/sft_coco_code_strict_v1'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    step0 = {}
    for row in rows:
        if row['id'].endswith('_step_00'):
            step0[row['meta']['trajectory_id']] = row
    judged_path = out / 'strict_judged.jsonl'
    done = {}
    if judged_path.exists():
        for line in judged_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record['id']] = record
    pending = [record for tid, record in step0.items() if tid not in done]
    if args.limit:
        pending = pending[:args.limit]
    client = make_client()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, record, client): record for record in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'id': record['meta']['trajectory_id'], 'pass': False,
                          'error': f'{type(exc).__name__}: {exc}'[:300]}
            done[result['id']] = result
            with judged_path.open('a') as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({k: result.get(k) for k in ('id', 'pass')}, ensure_ascii=False), flush=True)
    keep = {tid for tid, record in done.items() if record.get('pass')}
    kept_rows = [row for row in rows if row['meta']['trajectory_id'] in keep]
    train = [r for r in kept_rows if r['meta']['split'] == 'train']
    val = [r for r in kept_rows if r['meta']['split'] == 'val']
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (out / 'val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in val))
    summary = {'candidates': len(step0), 'judged': len(done), 'strict_pass': len(keep),
               'pass_rate': len(keep) / len(done) if done else 0,
               'train_rows': len(train), 'val_rows': len(val)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
