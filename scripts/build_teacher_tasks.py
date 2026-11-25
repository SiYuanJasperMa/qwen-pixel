"""Generate teacher reasoning/code for COCO editing tasks at scale."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_teacher_magicbrush import PROMPT, run_code  # noqa: E402
from build_teacher_multi import call_gpt55, call_traex, parse_json, traex_key  # noqa: E402

DEFAULT_TASKS = ROOT / 'qwen_pixel/data/edit_tasks_coco_v1/tasks.jsonl'


def proxy_key():
    path = ROOT / 'traex_api_proxy/proxy_logs/local.key'
    if path.exists():
        return path.read_text().strip()
    return traex_key()


def call_with_retry(fn, retries, label):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            status = getattr(getattr(exc, 'response', None), 'status_code', 0)
            if status and status not in (429, 500, 502, 503, 504):
                raise
            delay = min(300, 30 * (2 ** attempt))
            print(json.dumps({'retry': label, 'attempt': attempt + 1, 'delay': delay,
                              'error': type(exc).__name__, 'message': str(exc)[:200]}), flush=True)
            time.sleep(delay)
    raise last


def process_task(task, out, backend, key, retries):
    image_path = Path(task['image'])
    if not image_path.is_absolute():
        image_path = ROOT / 'qwen_image_edit_smoke' / image_path
    source = out / 'images' / f"{task['id']}_source.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    image.save(source)
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=task['instruction'])
    if backend == 'gpt55':
        text = call_with_retry(lambda: call_gpt55(prompt, b64), retries, task['id'])
    elif backend == 'terra':
        text = call_with_retry(lambda: call_traex('traex/GPT-5.6-Terra', prompt, b64, 600, key),
                               retries, task['id'])
    elif backend == 'luna':
        text = call_with_retry(lambda: call_traex('traex/GPT-5.6-Luna', prompt, b64, 600, key),
                               retries, task['id'])
    elif backend == 'sol':
        text = call_with_retry(lambda: call_traex('traex/GPT-5.6-Sol', prompt, b64, 900, key),
                               retries, task['id'])
    elif backend == 'gpt61':
        text = call_with_retry(lambda: call_traex('traex/GPT-6.1-Sol', prompt, b64, 1800, key),
                               retries, task['id'])
    elif backend == 'deepseek':
        text = call_with_retry(lambda: call_traex('traex/DeepSeek-V4.1-Flash', prompt, b64, 300, key),
                               retries, task['id'])
    else:
        raise ValueError(backend)
    parsed = parse_json(text)
    execution = run_code(parsed.get('executable_code', ''), source,
                         out / 'executions' / f"{task['id']}.png")
    return {
        'id': task['id'], 'task': 'coco_edit_code',
        'source_image': str(source), 'instruction': task['instruction'],
        'teacher_backend': backend,
        'reasoning': {k: parsed.get(k) for k in
                      ['visual_evidence', 'target_object', 'target_region',
                       'editing_intent', 'editing_plan', 'preservation_constraints']},
        'code': parsed.get('executable_code', ''),
        'requires_inpaint': parsed.get('requires_inpaint', False),
        'execution': execution,
        'meta': task.get('meta', {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['gpt55', 'terra', 'luna', 'sol', 'gpt61', 'deepseek'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--tasks', default=str(DEFAULT_TASKS))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / 'qwen_pixel' / out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    (out / 'executions').mkdir(exist_ok=True)
    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    tasks = tasks[args.start:args.start + args.limit if args.limit else None]
    done = {json.loads(line)['id'] for line in (out / 'samples.jsonl').read_text().splitlines() if line.strip()} \
        if (out / 'samples.jsonl').exists() else set()
    pending = [task for task in tasks if task['id'] not in done]
    key = traex_key() if args.backend != 'gpt55' else None
    print(f'pending={len(pending)} workers={args.workers} backend={args.backend}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_task, task, out, args.backend, key, args.retries): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                sample = future.result()
            except Exception as exc:
                print(json.dumps({'id': task['id'], 'error': type(exc).__name__,
                                  'message': str(exc)[:300]}), flush=True)
                continue
            with (out / 'samples.jsonl').open('a') as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(json.dumps({'id': sample['id'], 'backend': sample['teacher_backend'],
                              'requires_inpaint': sample['requires_inpaint'],
                              'execution': sample['execution']['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
