"""Convert accepted GPT-6.1-Sol MagicBrush inpaint trajectories to Qwen SFT rows.

Each row supervises one assistant action: the python pixel program, then finish.
The python program is re-executed in the student sandbox and must reproduce the
teacher's accepted RGB output exactly before the trajectory is kept.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import RGB_SYSTEM, RGB_TOOLS, execute_action  # noqa: E402


def compress_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


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


def build_one(sample, out, allow_mismatch=False, compress=False):
    sid = sample['id']
    source = Path(sample['source_image'])
    teacher_output = Path(sample['execution']['output'])
    workspace = out / 'executions' / sid
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copy2(source, workspace / 'source.png')

    variants = []
    if compress:
        try:
            variants.append(('compressed', compress_code(sample['code'])))
        except Exception:
            pass
    variants.append(('original', sample['code']))
    chosen = None
    failure = None
    for label, teacher_code in variants:
        code = action_code(teacher_code)
        feedback, ok = execute_action('python', {'code': code}, workspace, 0, rgb_profile=True)
        if not ok:
            failure = {'reason': f'{label} execution failed', 'feedback': feedback['content'][:500]}
            continue
        student_png = workspace / 'edited.png'
        if not student_png.exists():
            failure = {'reason': f'{label} produced no edited.png'}
            continue
        a = np.asarray(Image.open(student_png).convert('RGB'))
        b = np.asarray(Image.open(teacher_output).convert('RGB'))
        exact = a.shape == b.shape and np.array_equal(a, b)
        if not exact and not allow_mismatch:
            failure = {'reason': f'{label} native RGB mismatch', 'student_shape': list(a.shape),
                       'teacher_shape': list(b.shape),
                       'changed_pixels': int(np.count_nonzero(np.any(a != b, axis=-1))) if a.shape == b.shape else None}
            continue
        chosen = (label, teacher_code, code, feedback, exact)
        break
    if chosen is None:
        return None, failure or {'reason': 'no code variant replayed'}
    label, teacher_code, code, feedback, exact = chosen

    finish_args = {'image_path': 'edited.png', 'rgb_path': 'edited_rgb.npy'}
    finish_feedback, finish_ok = execute_action('finish', finish_args, workspace, 1)
    if not finish_ok:
        return None, {'reason': 'finish validation failed', 'feedback': finish_feedback['content'][:500]}

    width, height = Image.open(workspace / 'source.png').size
    system = {'role': 'system', 'content': RGB_SYSTEM}
    user = {'role': 'user', 'content': [
        {'type': 'image', 'image': str(workspace / 'source.png')},
        {'type': 'text', 'text': (f'Source file: source.png\nSource dimensions: width={width}, height={height}. '
                                  f'All editing coordinates must refer to this full source. '
                                  f'Map crop coordinates back to the source before editing.\n'
                                  f'EDIT: {sample["instruction"]}')}]}
    assistant_python = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'python', 'arguments': {'code': code}}}]}
    assistant_finish = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'finish', 'arguments': finish_args}}]}
    judge = sample.get('judge') or {}
    meta = {
        'trajectory_id': sid,
        'source': 'gpt61_magicbrush_inpaint_teacher',
        'instruction': sample['instruction'],
        'teacher_backend': sample.get('teacher_backend', 'gpt61'),
        'judge_scores': {k: judge.get(k) for k in
                         ('instruction_adherence', 'preservation', 'visual_quality', 'pass')},
        'teacher_code_sha256': hashlib.sha256(teacher_code.encode()).hexdigest(),
        'code_variant': label,
        'native_replay_exact': exact,
        'split': split_of(sid),
    }
    rows = [
        {'id': f'{sid}_step_00', 'task': 'agent_edit', 'messages': [system, user],
         'target_message': assistant_python, 'tools': RGB_TOOLS, 'meta': meta},
        {'id': f'{sid}_step_01', 'task': 'agent_edit',
         'messages': [system, user, assistant_python, feedback],
         'target_message': assistant_finish, 'tools': RGB_TOOLS, 'meta': meta},
    ]
    return rows, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judged', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--allow-mismatch', action='store_true')
    parser.add_argument('--compress', action='store_true',
                        help='try ast-unparsed compact code first; keep only exact replays')
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in Path(args.judged).read_text().splitlines() if line.strip()]
    samples = [s for s in samples if (s.get('judge') or {}).get('pass')]
    if args.limit:
        samples = samples[:args.limit]
    rows, skipped = [], []
    for sample in samples:
        built, reason = build_one(sample, out, args.allow_mismatch, args.compress)
        if built is None:
            skipped.append({'id': sample['id'], **reason})
            print(json.dumps(skipped[-1], ensure_ascii=False)[:300], flush=True)
            continue
        rows.extend(built)
        print(json.dumps({'kept': sample['id'], 'rows': len(built)}), flush=True)
    train = [r for r in rows if r['meta']['split'] == 'train']
    val = [r for r in rows if r['meta']['split'] == 'val']
    for name, subset in [('train', train), ('val', val)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
    manifest = {
        'accepted_teacher_trajectories': len(samples),
        'converted_trajectories': len(rows) // 2,
        'rows': len(rows),
        'train_rows': len(train),
        'val_rows': len(val),
        'skipped': skipped,
        'native_replay_exact_required': not args.allow_mismatch,
        'note': 'Teacher RGB re-executed in the student sandbox; only exact native replays kept.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('accepted_teacher_trajectories', 'converted_trajectories',
                                               'rows', 'train_rows', 'val_rows')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
