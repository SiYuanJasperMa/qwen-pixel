"""GPT-6.1-Sol MagicBrush teacher that must produce pixel-level inpaint code.

Unlike build_teacher_multi.py, this run does not allow the teacher to escape with
"requires_inpaint": every sample must return executable code. Classical pixel
inpainting (cv2.inpaint / patch synthesis) is allowed because it is not a learned
generator and the student can run the same code at inference time.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'qwen_pixel'))
from build_teacher_multi import (  # noqa: E402
    PARQUET, SYSTEM, call_gpt55, metrics, parse_json, save_resized, traex_key)
from pixel.teacher_api import judge_edit, make_client  # noqa: E402

TRAEX_URL = 'http://127.0.0.1:9584/v1/responses'


def call_traex_model(model, prompt, b64, key, timeout=1800, effort='high'):
    body = {
        'model': model,
        'input': [
            {'role': 'system', 'content': [{'type': 'input_text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [
                {'type': 'input_text', 'text': prompt},
                {'type': 'input_image', 'image_url': 'data:image/png;base64,' + b64}]},
        ],
        'max_output_tokens': 8192,
        'reasoning': {'effort': effort},
    }
    response = requests.post(TRAEX_URL, json=body, timeout=timeout,
                             headers={'Authorization': 'Bearer ' + key})
    response.raise_for_status()
    data = response.json()
    texts = []
    for message in data.get('output', []):
        for part in message.get('content', []):
            if part.get('type') in ('output_text', 'text'):
                texts.append(part.get('text', ''))
    return '\n'.join(texts).strip()


def call_sol(prompt, b64, key, timeout=1800, effort='high'):
    return call_traex_model('traex/GPT-6.1-Sol', prompt, b64, key, timeout, effort)


def call_teacher(backend, prompt, b64, key):
    if backend == 'gpt55':
        return call_gpt55(prompt, b64)
    if backend == 'gpt56sol':
        return call_traex_model('traex/GPT-5.6-Sol', prompt, b64, key)
    if backend == 'terra':
        return call_traex_model('traex/GPT-5.6-Terra', prompt, b64, key)
    if backend == 'luna':
        return call_traex_model('traex/GPT-5.6-Luna', prompt, b64, key)
    if backend == 'seed21pro':
        return call_traex_model('traex/Seed-2.1-Pro-0915', prompt, b64, key)
    if backend == 'seed21turbo':
        return call_traex_model('traex/Seed-2.1-Turbo', prompt, b64, key)
    if backend == 'deepseek':
        return call_traex_model('traex/DeepSeek-V4.1-Flash', prompt, b64, key, 300)
    return call_sol(prompt, b64, key)

PROMPT = '''TASK: RGB_TO_EDITING_CODE

INSTRUCTION:
{instruction}

You must directly edit the pixels by writing complete, runnable Python code.
Do NOT skip the task, do NOT return empty code, and do NOT say the edit is impossible.
If the instruction requires removing, replacing, or filling content, synthesize the new
pixels yourself with classical image processing:
- build an accurate mask for the affected region;
- for removal, fill from surrounding texture with cv2.inpaint or patch copying, then
  restore grain, noise and shading so the fill is not a gray smear;
- for replacement, construct the new object with layered shapes, perspective, shading,
  a cast shadow, and feathered edges; match the scene lighting, color statistics and noise;
- blend the boundary with cv2.seamlessClone or alpha feathering;
- never leave flat cartoon fills, visible polygon edges, or obvious blur patches.
You may read pixels from other parts of the same source image (patch synthesis), but not
from external files. Do not use any pretrained generative model, network access, or external files.

Return JSON with exactly these fields:
- visual_evidence: list of short visible facts that justify the edit
- target_object: the object or region to edit
- target_region: a short description of the target area
- editing_intent: what should change
- editing_plan: ordered list of steps
- preservation_constraints: list of things that must not change
- executable_code: a Python function `def edit(image):` that returns a PIL Image; it must be complete and runnable
- requires_inpaint: true if your code synthesizes new content, false otherwise

Available helpers: load_image(path), save_image(image, path), crop(image, box), resize(image, size),
recolor(image, mask, color), composite(source, edited, mask), preserve(source, edited, mask),
inpaint(image, mask, radius=3).
You may also use PIL, NumPy, and cv2 directly. Do not use the network, shell, or external files.
The source image is 512x512. Use `w, h = image.size` and relative coordinates; never hardcode a fixed resolution.
'''

RUNNER_PREAMBLE = '''import random
random.seed(0)
import numpy as np
np.random.seed(0)
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFilter
def load_image(path): return Image.open(path).convert("RGB")
def save_image(image, path): image.save(path)
def crop(image, box): return image.crop(box)
def resize(image, size): return image.resize(size, Image.Resampling.LANCZOS)
def recolor(image, mask, color):
    a=np.array(image).copy(); m=np.array(mask.convert("L"))>127; a[m]=color; return Image.fromarray(a)
def composite(source, edited, mask): return Image.composite(edited, source, mask.convert("L"))
def preserve(source, edited, mask): return composite(source, edited, mask)
def inpaint(image, mask, radius=3):
    arr=np.array(image.convert("RGB"))
    m=np.array(mask.convert("L")) if hasattr(mask,"convert") else np.asarray(mask)
    m=(m>127).astype(np.uint8)*255
    return Image.fromarray(cv2.inpaint(arr, m, int(radius), cv2.INPAINT_TELEA))
'''


def run_code(code, source_path, out_path, timeout=60):
    if not code.strip():
        return {'status': 'skipped', 'error': 'empty code', 'output': None}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input.png').write_bytes(source_path.read_bytes())
        script = tmp / 'run.py'
        script.write_text(
            RUNNER_PREAMBLE + code + '\n'
            'if __name__ == "__main__":\n'
            '    img=load_image("input.png")\n'
            '    out=edit(img)\n'
            '    out.save("output.png")\n'
        )
        try:
            result = subprocess.run([sys.executable, str(script)], cwd=tmp, text=True,
                                    capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'timeout', 'output': None}
        if result.returncode != 0:
            return {'status': 'failed', 'error': (result.stderr or '')[-800:], 'output': None}
        if not (tmp / 'output.png').exists():
            return {'status': 'failed', 'error': 'no output.png', 'output': None}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes((tmp / 'output.png').read_bytes())
        return {'status': 'passed', 'error': '', 'output': str(out_path)}


def process_row(row, out, key, repair_attempts=2, backend='gpt61'):
    sample_id = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
    source = out / 'images' / f'{sample_id}_source.png'
    target = out / 'images' / f'{sample_id}_target.png'
    mask = out / 'images' / f'{sample_id}_mask.png'
    save_resized(row['source_img']['bytes'], source, 'RGB')
    save_resized(row['target_img']['bytes'], target, 'RGB')
    save_resized(row['mask_img']['bytes'], mask, 'L')
    b64 = base64.b64encode(source.read_bytes()).decode()
    base_prompt = PROMPT.format(instruction=row['instruction'])
    prompt = base_prompt
    t0 = time.time()
    parsed = {}
    code = ''
    last_text = ''
    execution = {'status': 'skipped', 'error': 'no attempt', 'output': None}
    attempts = []
    for attempt in range(1, repair_attempts + 2):
        text = call_teacher(backend, prompt, b64, key)
        last_text = text
        (out / 'raw' / f'{sample_id}_attempt{attempt}.txt').write_text(text)
        try:
            parsed = parse_json(text)
            code = parsed.get('executable_code', '') or ''
            execution = run_code(code, source, out / 'executions' / f'{sample_id}.png')
        except Exception as exc:
            execution = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'output': None}
        attempts.append({'attempt': attempt, 'code': code,
                         'execution': execution, 'raw_text': text[:6000]})
        if execution['status'] == 'passed':
            break
        prompt = base_prompt + (
            f"\n\nYour previous attempt returned:\n{text[:6000]}\n\n"
            f"Execution failed with:\n{str(execution.get('error'))[:800]}\n"
            "Return a corrected complete JSON object with the same schema. "
            "Fix the bug; do not repeat it."
        )
    elapsed = round(time.time() - t0, 1)
    m = metrics(source, target, out / 'executions' / f'{sample_id}.png', mask) \
        if execution['status'] == 'passed' else {}
    return {
        'id': sample_id, 'task': 'rgb_to_editing_code_inpaint',
        'source_image': str(source), 'target_image': str(target), 'mask_image': str(mask),
        'instruction': row['instruction'], 'teacher_backend': backend,
        'teacher_elapsed_seconds': elapsed,
        'reasoning': {k: parsed.get(k) for k in
                      ['visual_evidence', 'target_object', 'target_region',
                       'editing_intent', 'editing_plan', 'preservation_constraints']},
        'code': code,
        'requires_inpaint': parsed.get('requires_inpaint', False),
        'execution': execution, 'metrics': m, 'attempts': attempts,
    }


def judge_sample(sample, client):
    if sample['execution']['status'] != 'passed':
        return {'skipped': True, 'reason': 'no executed image'}
    judgment = judge_edit(client, sample['source_image'], sample['execution']['output'],
                          sample['instruction'], strict_extent=False)
    return judgment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ids-from', default='data/teacher_gpt55_100_v1/samples.jsonl',
                        help='JSONL whose requires_inpaint=true ids seed this run')
    parser.add_argument('--include-control', type=int, default=0,
                        help='extra non-inpaint samples as control')
    parser.add_argument('--repair-attempts', type=int, default=2,
                        help='extra teacher attempts after an execution failure')
    parser.add_argument('--backend', choices=['gpt61', 'gpt55', 'terra', 'luna'], default='gpt61')
    parser.add_argument('--judge-workers', type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / 'qwen_pixel' / out
    out.mkdir(parents=True, exist_ok=True)
    for sub in ['raw', 'images', 'executions']:
        (out / sub).mkdir(exist_ok=True)
    samples_path = out / 'samples.jsonl'
    if not samples_path.exists():
        samples_path.write_text('')
    done = {json.loads(line)['id'] for line in samples_path.read_text().splitlines() if line.strip()}

    seed_rows = [json.loads(line) for line in (ROOT / 'qwen_pixel' / args.ids_from).read_text().splitlines() if line.strip()]
    inpaint_ids = [r['id'] for r in seed_rows if r.get('requires_inpaint')]
    control_ids = [r['id'] for r in seed_rows if not r.get('requires_inpaint')]
    wanted = inpaint_ids[:args.limit] + control_ids[:args.include_control]
    wanted = [i for i in wanted if i not in done]
    print(f'pending={len(wanted)} workers={args.workers} inpaint={len(inpaint_ids)}', flush=True)

    wanted_set = set(wanted)
    table = pq.read_table(PARQUET, columns=['img_id', 'turn_index', 'instruction',
                                            'source_img', 'target_img', 'mask_img'])
    rows = []
    for i in range(table.num_rows):
        row = {k: table[k][i].as_py() for k in table.column_names}
        sid = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
        if sid in wanted_set:
            rows.append(row)
    print(f'matched={len(rows)}', flush=True)

    key = traex_key() if args.backend != 'gpt55' else None
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, row, out, key, args.repair_attempts, args.backend): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            sid = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
            try:
                sample = future.result()
            except Exception as exc:
                print(json.dumps({'id': sid, 'error': type(exc).__name__,
                                  'message': str(exc)[:400]}), flush=True)
                continue
            with samples_path.open('a') as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            results.append(sample)
            print(json.dumps({'id': sid, 'elapsed': sample['teacher_elapsed_seconds'],
                              'requires_inpaint': sample['requires_inpaint'],
                              'execution': sample['execution']['status'],
                              'metrics': sample['metrics']}, ensure_ascii=False), flush=True)

    client = make_client()
    judged_by_id = {}
    if (out / 'judged.jsonl').exists():
        for line in (out / 'judged.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                judged_by_id[row['id']] = row
    def judge_and_attach(sample):
        try:
            return sample, judge_sample(sample, client)
        except Exception as exc:
            return sample, {'error': type(exc).__name__, 'message': str(exc)[:300]}

    pending = [sample for sample in results if sample['id'] not in judged_by_id]
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        futures = [pool.submit(judge_and_attach, sample) for sample in pending]
        for future in as_completed(futures):
            sample, judgment = future.result()
            sample['judge'] = judgment
            judged_by_id[sample['id']] = sample
            print(json.dumps({'judged': sample['id'], 'pass': bool(judgment.get('pass'))}), flush=True)
    all_samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    merged = [judged_by_id.get(sample['id'], sample) for sample in all_samples]
    if merged:
        (out / 'judged.jsonl').write_text(
            ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in merged))
    executed = sum(1 for s in merged if s['execution']['status'] == 'passed')
    passed = sum(1 for s in merged if (s.get('judge') or {}).get('pass'))
    repaired = sum(1 for s in merged if len(s.get('attempts') or []) > 1)
    summary = {
        'teacher': args.backend,
        'tasks': len(merged),
        'execution_passed': executed,
        'judge_pass': passed,
        'overall_pass_rate': passed / len(merged) if merged else 0,
        'executed_pass_rate': passed / executed if executed else 0,
        'repair_used': repaired,
        'control_samples': args.include_control,
        'source_ids': args.ids_from,
        'note': 'MagicBrush train samples previously flagged requires_inpaint; teacher forced to produce classical pixel inpaint code.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
