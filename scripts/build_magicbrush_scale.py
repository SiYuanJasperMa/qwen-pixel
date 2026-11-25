"""Scale MagicBrush teacher data construction toward 10k SFT rows.

For each MagicBrush row, try teachers in priority order (GPT-6.1-Sol, Terra,
Luna, GPT-5.5). Execute the returned pixel code, judge the actual RGB result,
and keep the first accepted trajectory. Accepted trajectories are appended to
`judged.jsonl`; every attempt is recorded in `attempts.jsonl` for resume.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from build_teacher_magicbrush_inpaint import (  # noqa: E402
    PROMPT, call_teacher, judge_sample, metrics, parse_json, run_code, save_resized)
from build_teacher_multi import traex_key  # noqa: E402
from pixel.teacher_api import make_client  # noqa: E402

DEFAULT_SHARDS = '/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke/data/magicbrush_hf/data/train-*.parquet'
LOCK = threading.Lock()


def load_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()}


def load_attempts(path):
    pairs = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                pairs.add((row['id'], row['teacher']))
    return pairs


def append_jsonl(path, record):
    with LOCK:
        with path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def process_sample(row, out, teachers, key, client, attempted):
    sid = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
    source = out / 'images' / f'{sid}_source.png'
    target = out / 'images' / f'{sid}_target.png'
    mask = out / 'images' / f'{sid}_mask.png'
    if not source.exists():
        save_resized(row['source_img']['bytes'], source, 'RGB')
    if not target.exists():
        save_resized(row['target_img']['bytes'], target, 'RGB')
    if not mask.exists():
        save_resized(row['mask_img']['bytes'], mask, 'L')
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=row['instruction'])
    for teacher in teachers:
        if (sid, teacher) in attempted:
            continue
        t0 = time.time()
        record = {'id': sid, 'teacher': teacher, 'instruction': row['instruction'],
                  'source_image': str(source), 'target_image': str(target),
                  'mask_image': str(mask), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            text = call_teacher(teacher, prompt, b64, key)
            (out / 'raw' / f'{sid}__{teacher}.txt').write_text(text)
            parsed = parse_json(text)
            code = parsed.get('executable_code', '') or ''
            execution = run_code(code, source, out / 'executions' / f'{sid}__{teacher}.png')
        except Exception as exc:
            record.update({'status': 'teacher_error', 'error': f'{type(exc).__name__}: {exc}'[:400],
                           'elapsed_seconds': round(time.time() - t0, 1)})
            append_jsonl(out / 'attempts.jsonl', record)
            continue
        record.update({'status': execution['status'], 'elapsed_seconds': round(time.time() - t0, 1),
                       'code': code, 'execution': execution,
                       'reasoning': {k: parsed.get(k) for k in
                                     ['visual_evidence', 'target_object', 'target_region',
                                      'editing_intent', 'editing_plan', 'preservation_constraints']},
                       'requires_inpaint': parsed.get('requires_inpaint', False)})
        if execution['status'] == 'passed':
            m = metrics(source, target, out / 'executions' / f'{sid}__{teacher}.png', mask)
            record['metrics'] = m
            try:
                judgment = judge_sample({'source_image': str(source),
                                         'execution': execution,
                                         'instruction': row['instruction']}, client)
            except Exception as exc:
                judgment = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            record['judge'] = judgment
            if judgment.get('pass'):
                record['status'] = 'pass'
                append_jsonl(out / 'attempts.jsonl', record)
                accepted = {'id': sid, 'task': 'rgb_to_editing_code_inpaint',
                            'source_image': str(source), 'target_image': str(target),
                            'mask_image': str(mask), 'instruction': row['instruction'],
                            'teacher_backend': teacher,
                            'teacher_elapsed_seconds': record['elapsed_seconds'],
                            'reasoning': record['reasoning'], 'code': code,
                            'requires_inpaint': record['requires_inpaint'],
                            'execution': execution, 'metrics': m, 'judge': judgment}
                append_jsonl(out / 'judged.jsonl', accepted)
                return {'id': sid, 'accepted': True, 'teacher': teacher, 'judge': judgment}
            record['status'] = 'judge_fail'
        append_jsonl(out / 'attempts.jsonl', record)
    return {'id': sid, 'accepted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', default=DEFAULT_SHARDS)
    parser.add_argument('--out', default=str(ROOT / 'data/teacher_magicbrush_scale_v1'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--teachers', default='gpt61,terra,luna,gpt55')
    parser.add_argument('--target-trajectories', type=int, default=5000)
    parser.add_argument('--limit-samples', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    for sub in ['raw', 'images', 'executions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    accepted = load_ids(out / 'judged.jsonl')
    attempted = load_attempts(out / 'attempts.jsonl')
    teachers = [t.strip() for t in args.teachers.split(',') if t.strip()]
    key = traex_key()
    client = make_client()
    shards = sorted(glob.glob(args.shards))
    print(f'shards={len(shards)} accepted={len(accepted)} attempted_pairs={len(attempted)} '
          f'teachers={teachers} target={args.target_trajectories}', flush=True)
    scheduled = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard in shards:
            if len(accepted) >= args.target_trajectories:
                break
            table = pq.read_table(shard, columns=['img_id', 'turn_index', 'instruction',
                                                  'source_img', 'target_img', 'mask_img'])
            futures = []
            for i in range(table.num_rows):
                if args.limit_samples and scheduled >= args.limit_samples:
                    break
                if len(accepted) >= args.target_trajectories:
                    break
                row = {k: table[k][i].as_py() for k in table.column_names}
                sid = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
                if sid in accepted:
                    continue
                if all((sid, teacher) in attempted for teacher in teachers):
                    continue
                futures.append(pool.submit(process_sample, row, out, teachers, key, client, attempted))
                scheduled += 1
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:300]}), flush=True)
                    continue
                if result.get('accepted'):
                    with LOCK:
                        accepted.add(result['id'])
                    print(json.dumps({'accepted': result['id'], 'teacher': result['teacher'],
                                      'total_accepted': len(accepted),
                                      'sft_rows_est': len(accepted) * 2}), flush=True)
                if len(accepted) >= args.target_trajectories:
                    break
    print(json.dumps({'accepted_trajectories': len(accepted), 'sft_rows_est': len(accepted) * 2,
                      'attempted_pairs': len(load_attempts(out / 'attempts.jsonl')),
                      'target_trajectories': args.target_trajectories}, indent=2), flush=True)


if __name__ == '__main__':
    main()
