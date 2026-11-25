"""Judge and convert execution-passed COCO edit code into Qwen agent SFT rows.

The source dataset only guarantees that the code runs. This script re-executes
each program in the student sandbox, judges the actual RGB result, and keeps only
semantic passes as python -> finish trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import RGB_SYSTEM, RGB_TOOLS, execute_action  # noqa: E402
from pixel.teacher_api import judge_edit, make_client  # noqa: E402


def action_code(teacher_code):
    return teacher_code + '''

_out = edit(image)
_arr = np.asarray(_out)
if _arr.ndim == 2:
    _arr = np.stack([_arr, _arr, _arr], axis=-1)
emit_rgb(_arr)
'''


def split_of(sample_id):
    return 'val' if int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'


def build_one(row, out, client):
    sid = row['id']
    workspace = out / 'executions' / sid
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copy2(row['image'], workspace / 'source.png')
    code = action_code(row['answer'])
    feedback, ok = execute_action('python', {'code': code}, workspace, 0, rgb_profile=True)
    if not ok:
        return None, {'reason': 'execution failed', 'feedback': feedback['content'][:400]}
    finish_args = {'image_path': 'edited.png', 'rgb_path': 'edited_rgb.npy'}
    finish_feedback, finish_ok = execute_action('finish', finish_args, workspace, 1)
    if not finish_ok:
        return None, {'reason': 'finish failed', 'feedback': finish_feedback['content'][:400]}
    try:
        judgment = judge_edit(client, str(workspace / 'source.png'),
                              str(workspace / 'edited.png'), row['instruction'],
                              strict_extent=False)
    except Exception as exc:
        return None, {'reason': 'judge error', 'error': f'{type(exc).__name__}: {exc}'[:300]}
    if not judgment.get('pass'):
        return None, {'reason': 'judge fail', 'judge': judgment}
    width, height = Image.open(workspace / 'source.png').size
    system = {'role': 'system', 'content': RGB_SYSTEM}
    user = {'role': 'user', 'content': [
        {'type': 'image', 'image': str(workspace / 'source.png')},
        {'type': 'text', 'text': (f'Source file: source.png\nSource dimensions: width={width}, height={height}. '
                                  f'All editing coordinates must refer to this full source. '
                                  f'Map crop coordinates back to the source before editing.\n'
                                  f'EDIT: {row["instruction"]}')}]}
    assistant_python = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'python', 'arguments': {'code': code}}}]}
    assistant_finish = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'finish', 'arguments': finish_args}}]}
    meta = {
        'trajectory_id': sid,
        'source': 'coco_edit_teacher_judged',
        'instruction': row['instruction'],
        'teacher_backend': (row.get('meta') or {}).get('teacher_backend'),
        'judge_scores': {k: judgment.get(k) for k in
                         ('instruction_adherence', 'preservation', 'visual_quality', 'pass')},
        'code_sha256': hashlib.sha256(row['answer'].encode()).hexdigest(),
        'split': split_of(sid),
    }
    return [
        {'id': f'{sid}_step_00', 'task': 'agent_edit', 'messages': [system, user],
         'target_message': assistant_python, 'tools': RGB_TOOLS, 'meta': meta},
        {'id': f'{sid}_step_01', 'task': 'agent_edit',
         'messages': [system, user, assistant_python, feedback],
         'target_message': assistant_finish, 'tools': RGB_TOOLS, 'meta': meta},
    ], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data/sft_code_coco_v1/train.jsonl'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    judged_path = out / 'judged.jsonl'
    done = set()
    if judged_path.exists():
        done = {json.loads(line)['id'] for line in judged_path.read_text().splitlines() if line.strip()}
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['id'] not in done]
    if args.limit:
        rows = rows[:args.limit]
    client = make_client()
    rows_path = out / 'rows.jsonl'
    kept = []
    if rows_path.exists():
        kept = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    skipped = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, row, out, client): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                built, reason = future.result()
            except Exception as exc:
                built, reason = None, {'reason': 'unhandled', 'error': f'{type(exc).__name__}: {exc}'[:300]}
            record = {'id': row['id'], 'kept': built is not None}
            if built is None:
                record['reason'] = reason
                skipped.append(record)
                print(json.dumps(record, ensure_ascii=False)[:300], flush=True)
                continue
            kept.extend(built)
            with rows_path.open('a') as handle:
                for built_row in built:
                    handle.write(json.dumps(built_row, ensure_ascii=False) + '\n')
            record['judge'] = built[0]['meta']['judge_scores']
            with judged_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
    train = [r for r in kept if r['meta']['split'] == 'train']
    val = [r for r in kept if r['meta']['split'] == 'val']
    for name, subset in [('train', train), ('val', val)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
    manifest = {'input': args.data, 'converted_rows': len(kept), 'train_rows': len(train),
                'val_rows': len(val), 'kept_trajectories': len(kept) // 2,
                'skipped_this_run': len(skipped)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
