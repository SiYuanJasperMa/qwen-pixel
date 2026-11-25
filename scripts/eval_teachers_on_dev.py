"""Evaluate teacher models one-shot on the three editing dev benches.

The teacher sees exactly the same system/prompt as the student SFT rows: source
image + instruction, must return JSON with executable_code. Code is executed and
judged by the same gpt-5.5 judge used for student evals.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import (  # noqa: E402
    PROMPT, call_teacher, call_traex_model, parse_json, run_code)
from build_teacher_multi import traex_key  # noqa: E402
from pixel.teacher_api import judge_edit, make_client  # noqa: E402

BENCHES = {
    'magicbrush': ROOT / 'data/magicbrush_dev32_v1/tasks.jsonl',
    'ultraedit': ROOT / 'data/sft_ultraedit_oneshot_full_v1/dev_tasks.jsonl',
    'sciforma': ROOT / 'data/sft_sciforma_oneshot_full_v1/dev_tasks.jsonl',
}

TEACHERS = {
    'gpt61': lambda p, b, k: call_teacher('gpt61', p, b, k),
    'gpt56sol': lambda p, b, k: call_traex_model('traex/GPT-5.6-Sol', p, b, k, 1200),
    'gpt55': lambda p, b, k: call_teacher('gpt55', p, b, k),
    'terra': lambda p, b, k: call_teacher('terra', p, b, k),
    'luna': lambda p, b, k: call_teacher('luna', p, b, k),
    'deepseek': lambda p, b, k: call_traex_model('traex/DeepSeek-V4.1-Flash', p, b, k, 300),
    'seed21pro': lambda p, b, k: call_teacher('seed21pro', p, b, k),
    'seed21turbo': lambda p, b, k: call_teacher('seed21turbo', p, b, k),
}


def call_with_retry(fn, attempts=4):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            msg = str(exc)
            if any(tag in msg for tag in ('4055', 'Too many requests', '502', '429', 'timed out')):
                time.sleep(10 * (i + 1))
                continue
            raise
    raise last


def process(task, out, key, client, teacher):
    source = Path(task['image'])
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=task['instruction'])
    t0 = time.time()
    record = {'id': task['id'], 'bench': task['bench'], 'teacher': teacher,
              'instruction': task['instruction'], 'source_image': str(source)}
    try:
        text = call_with_retry(lambda: TEACHERS[teacher](prompt, b64, key))
        (out / 'raw' / f"{task['id']}.txt").write_text(text)
        parsed = parse_json(text)
        code = parsed.get('executable_code', '') or ''
        execution = run_code(code, source, out / 'executions' / f"{task['id']}.png")
    except Exception as exc:
        record.update({'status': 'teacher_error', 'error': f'{type(exc).__name__}: {exc}'[:400],
                       'elapsed_seconds': round(time.time() - t0, 1)})
        return record
    record.update({'status': execution['status'], 'code': code, 'execution': execution,
                   'elapsed_seconds': round(time.time() - t0, 1),
                   'reasoning': {k: parsed.get(k) for k in
                                 ['visual_evidence', 'target_object', 'target_region',
                                  'editing_intent', 'editing_plan', 'preservation_constraints']}})
    if execution['status'] == 'passed':
        try:
            record['judge'] = judge_edit(client, str(source), execution['output'],
                                         task['instruction'], strict_extent=False)
        except Exception as exc:
            record['judge'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        if (record.get('judge') or {}).get('pass'):
            record['status'] = 'pass'
    return record


def summarize(rows):
    out = {}
    for bench in BENCHES:
        part = [r for r in rows if r['bench'] == bench]
        executed = sum(1 for r in part if r['status'] in ('pass', 'passed', 'judge_fail'))
        passed = sum(1 for r in part if (r.get('judge') or {}).get('pass'))
        out[bench] = {'tasks': len(part), 'executed': executed, 'judge_pass': passed,
                      'overall_pass_rate': passed / len(part) if part else 0,
                      'executed_pass_rate': passed / executed if executed else 0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True, choices=list(TEACHERS))
    parser.add_argument('--benches', default='magicbrush,ultraedit,sciforma')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    benches = [b.strip() for b in args.benches.split(',') if b.strip()]
    out = Path(args.out or ROOT / 'data' / f'teacher_eval_{args.teacher}')
    for sub in ['raw', 'executions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    tasks = []
    for bench in benches:
        rows = [json.loads(line) for line in BENCHES[bench].read_text().splitlines() if line.strip()]
        for row in rows:
            tasks.append({'id': row['id'], 'image': row['image'],
                          'instruction': row['instruction'], 'bench': bench})
    judged_path = out / 'judged.jsonl'
    done = set()
    if judged_path.exists():
        for line in judged_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('status') != 'teacher_error':
                done.add(row['id'])
    tasks = [t for t in tasks if t['id'] not in done]
    if args.limit:
        tasks = tasks[:args.limit]
    print(json.dumps({'teacher': args.teacher, 'pending': len(tasks), 'done': len(done)}), flush=True)
    key = traex_key()
    client = make_client()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, t, out, key, client, args.teacher): t for t in tasks}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                print(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:300]}), flush=True)
                continue
            results.append(record)
            with judged_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'id': record['id'], 'bench': record['bench'],
                              'exec': record['status'],
                              'pass': (record.get('judge') or {}).get('pass')}), flush=True)
    all_rows = [json.loads(line) for line in judged_path.read_text().splitlines() if line.strip()]
    summary = summarize(all_rows)
    (out / 'summary.json').write_text(json.dumps({'teacher': args.teacher, 'benches': summary}, indent=2))
    print(json.dumps({'teacher': args.teacher, 'benches': summary}, indent=2))


if __name__ == '__main__':
    main()
