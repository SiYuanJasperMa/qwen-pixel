"""Generate additional MagicBrush trajectories by running other teachers on accepted samples."""
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
    PROMPT, call_teacher, judge_sample, metrics, parse_json, run_code)
from build_teacher_multi import traex_key  # noqa: E402
from pixel.teacher_api import make_client  # noqa: E402

TEACHERS = ['gpt61', 'terra', 'luna', 'gpt55']


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def process(sample, out, key, client):
    sid = sample['id']
    source = Path(sample['source_image'])
    target = Path(sample['target_image'])
    mask = Path(sample['mask_image'])
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=sample['instruction'])
    accepted_teacher = sample.get('teacher_backend')
    for teacher in TEACHERS:
        if teacher == accepted_teacher:
            continue
        t0 = time.time()
        record = {'id': f'{sid}__{teacher}', 'sample_id': sid, 'teacher': teacher,
                  'instruction': sample['instruction'],
                  'source_image': str(source), 'target_image': str(target),
                  'mask_image': str(mask)}
        try:
            text = call_teacher(teacher, prompt, b64, key)
            (out / 'raw' / f'{sid}__{teacher}.txt').write_text(text)
            parsed = parse_json(text)
            code = parsed.get('executable_code', '') or ''
            execution = run_code(code, source, out / 'executions' / f'{sid}__{teacher}.png')
        except Exception as exc:
            record.update({'status': 'teacher_error', 'error': f'{type(exc).__name__}: {exc}'[:300]})
            continue
        record.update({'status': execution['status'], 'code': code, 'execution': execution,
                       'elapsed_seconds': round(time.time() - t0, 1),
                       'reasoning': {k: parsed.get(k) for k in
                                     ['visual_evidence', 'target_object', 'target_region',
                                      'editing_intent', 'editing_plan', 'preservation_constraints']}})
        if execution['status'] == 'passed':
            m = metrics(source, target, execution['output'], mask)
            record['metrics'] = m
            try:
                judgment = judge_sample({'source_image': str(source), 'execution': execution,
                                         'instruction': sample['instruction']}, client)
            except Exception as exc:
                judgment = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            record['judge'] = judgment
            if judgment.get('pass'):
                record['status'] = 'pass'
                return record
            record['status'] = 'judge_fail'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accepted', default=str(ROOT / 'data/teacher_magicbrush_scale_v1/judged.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'data/teacher_magicbrush_variants_v1'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--target', type=int, default=5000)
    args = parser.parse_args()
    out = Path(args.out)
    for sub in ['raw', 'executions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    accepted = load_jsonl(Path(args.accepted))
    done = {r['sample_id'] for r in load_jsonl(out / 'variants.jsonl')}
    pending = [r for r in accepted if r['id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f'accepted={len(accepted)} done={len(done)} pending={len(pending)} target={args.target}', flush=True)
    key = traex_key()
    client = make_client()
    kept = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, sample, out, key, client): sample for sample in pending}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                print(json.dumps({'id': sample['id'], 'error': type(exc).__name__,
                                  'message': str(exc)[:300]}), flush=True)
                continue
            if record is None:
                continue
            with (out / 'variants.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
            print(json.dumps({'variant': record['id'], 'teacher': record['teacher'],
                              'total_variants': kept}), flush=True)
            if kept >= args.target:
                break
    print(json.dumps({'new_variants': kept, 'target': args.target}, indent=2), flush=True)


if __name__ == '__main__':
    main()
