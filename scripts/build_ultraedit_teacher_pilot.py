"""UltraEdit traditional-editing teacher pilot (GPT-6.1-Sol writes pixel code)."""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import PROMPT, call_teacher, parse_json, run_code  # noqa: E402
from build_teacher_multi import save_resized, traex_key  # noqa: E402
from pixel.teacher_api import judge_edit, make_client  # noqa: E402

PARQUET = str(ROOT / 'data/external_edit/UltraEdit_Region_Based_100k/data/RegionBase-{shard:05d}-of-00093.parquet')


def process(row, out, key, client, teacher='gpt61'):
    sid = f"ultraedit_{row['idx']}"
    source = out / 'images' / f'{sid}_source.png'
    target = out / 'images' / f'{sid}_target.png'
    mask = out / 'images' / f'{sid}_mask.png'
    save_resized(row['source_image']['bytes'], source, 'RGB')
    save_resized(row['edited_image']['bytes'], target, 'RGB')
    save_resized(row['mask_image']['bytes'], mask, 'L')
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=row['edit_prompt'])
    t0 = time.time()
    record = {'id': sid, 'instruction': row['edit_prompt'], 'edit_object': row['edit_object'],
              'source_image': str(source), 'target_image': str(target), 'mask_image': str(mask),
              'teacher_backend': teacher, 'generation_mode': 'rft'}
    try:
        text = call_teacher(teacher, prompt, b64, key)
        parsed = parse_json(text)
        code = parsed.get('executable_code', '') or ''
        execution = run_code(code, source, out / 'executions' / f'{sid}.png')
    except Exception as exc:
        record.update({'status': 'teacher_error', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        return record
    record.update({'status': execution['status'], 'code': code, 'execution': execution,
                   'requires_inpaint': parsed.get('requires_inpaint', False),
                   'elapsed_seconds': round(time.time() - t0, 1),
                   'reasoning': {k: parsed.get(k) for k in
                                 ['visual_evidence', 'target_object', 'target_region',
                                  'editing_intent', 'editing_plan', 'preservation_constraints']}})
    if execution['status'] == 'passed':
        try:
            record['judge'] = judge_edit(client, str(source), execution['output'],
                                         row['edit_prompt'], strict_extent=False)
        except Exception as exc:
            record['judge'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        if (record.get('judge') or {}).get('pass'):
            record['status'] = 'pass'
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shard', type=int, default=5)
    parser.add_argument('--teacher', default='gpt61', choices=['gpt61', 'terra', 'luna', 'gpt55', 'seed21pro', 'seed21turbo', 'deepseek', 'gpt56sol'])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    out = Path(args.out or ROOT / 'data' / f'ultraedit_teacher_shard{args.shard}_v1')
    for sub in ['images', 'executions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    table = pq.read_table(PARQUET.format(shard=args.shard))
    rows = [{k: table[k][i].as_py() for k in table.column_names} for i in range(table.num_rows)]
    done = set()
    judged_path = out / 'judged.jsonl'
    if judged_path.exists():
        done = {json.loads(line)['id'] for line in judged_path.read_text().splitlines() if line.strip()}
    rows = [r for r in rows if f"ultraedit_{r['idx']}" not in done]
    random.seed(0)
    random.shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    key = traex_key()
    client = make_client()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row, out, key, client, args.teacher): row for row in rows}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                print(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:300]}), flush=True)
                continue
            results.append(record)
            with (out / 'judged.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'id': record['id'], 'exec': record['status'],
                              'pass': (record.get('judge') or {}).get('pass')}), flush=True)
    all_rows = [json.loads(line) for line in judged_path.read_text().splitlines() if line.strip()] if judged_path.exists() else results
    executed = sum(1 for r in all_rows if r['status'] in ('pass', 'passed', 'judge_fail'))
    passed = sum(1 for r in all_rows if (r.get('judge') or {}).get('pass'))
    summary = {'tasks': len(all_rows), 'executed': executed, 'judge_pass': passed,
               'overall_pass_rate': passed / len(all_rows) if all_rows else 0,
               'executed_pass_rate': passed / executed if executed else 0}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
