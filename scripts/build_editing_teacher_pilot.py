"""Teacher pilots for SciForma, AnyEdit, and MaskEdit editing datasets."""
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

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import PROMPT, call_teacher, parse_json, run_code  # noqa: E402
from build_teacher_multi import save_resized, traex_key  # noqa: E402
from pixel.teacher_api import judge_edit, make_client  # noqa: E402

SOURCES = {
    'sciforma': ('data/external_edit/SciFormaData-700K/data/editing/train-{shard:05d}-of-00089.parquet',
                 'source_image', 'target_image', 'caption', None),
    'anyedit': ('data/external_edit/AnyEdit/data/train-00000-of-00383.parquet',
                'image_file', 'edited_file', 'edit_instruction', None),
    'maskedit': ('data/external_edit/MaskEdit-10k/data/infographics_cn/train-00000-of-00021.parquet',
                 'source', 'target', 'edit_instruction', 'mask'),
}


def image_bytes(value):
    if isinstance(value, dict):
        return value.get('bytes')
    return value


def process(item, out, key, client, teacher='gpt61'):
    sid, source_bytes, target_bytes, instruction, mask_bytes = item
    source = out / 'images' / f'{sid}_source.png'
    target = out / 'images' / f'{sid}_target.png'
    save_resized(source_bytes, source, 'RGB')
    save_resized(target_bytes, target, 'RGB')
    if mask_bytes:
        save_resized(mask_bytes, out / 'images' / f'{sid}_mask.png', 'L')
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=instruction)
    t0 = time.time()
    record = {'id': sid, 'instruction': instruction, 'source_image': str(source),
              'target_image': str(target), 'teacher_backend': teacher, 'generation_mode': 'rft'}
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
            record['judge'] = judge_edit(client, str(source), execution['output'], instruction,
                                         strict_extent=False)
        except Exception as exc:
            record['judge'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        if (record.get('judge') or {}).get('pass'):
            record['status'] = 'pass'
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(SOURCES))
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--teacher', default='gpt61', choices=['gpt61', 'terra', 'luna', 'gpt55', 'seed21pro', 'seed21turbo', 'deepseek', 'gpt56sol'])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    rel, src_col, tgt_col, ins_col, mask_col = SOURCES[args.dataset]
    rel = rel.format(shard=args.shard)
    out = Path(args.out or ROOT / 'data' / f'{args.dataset}_teacher_pilot_v1')
    for sub in ['images', 'executions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    table = pq.read_table(ROOT / rel)
    rows = []
    for i in range(table.num_rows):
        source = image_bytes(table[src_col][i].as_py())
        target = image_bytes(table[tgt_col][i].as_py())
        instruction = table[ins_col][i].as_py()
        if not source or not target or not instruction:
            continue
        mask = image_bytes(table[mask_col][i].as_py()) if mask_col else None
        sid = f'{args.dataset}_s{args.shard}_{i}' if args.dataset == 'sciforma' else f'{args.dataset}_{i}'
        rows.append((sid, source, target, instruction, mask))
    judged_path = out / 'judged.jsonl'
    done = {json.loads(line)['id'] for line in judged_path.read_text().splitlines() if line.strip()} if judged_path.exists() else set()
    rows = [r for r in rows if r[0] not in done]
    random.seed(0)
    random.shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    key = traex_key()
    client = make_client()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item, out, key, client, args.teacher): item for item in rows}
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
    executed = sum(1 for r in results if r['status'] in ('pass', 'passed', 'judge_fail'))
    passed = sum(1 for r in results if (r.get('judge') or {}).get('pass'))
    summary = {'dataset': args.dataset, 'tasks': len(results), 'executed': executed,
               'judge_pass': passed,
               'overall_pass_rate': passed / len(results) if results else 0,
               'executed_pass_rate': passed / executed if executed else 0}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
