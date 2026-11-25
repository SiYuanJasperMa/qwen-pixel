"""Run GPT-5.5, DeepSeek-V4.1-Flash, or GPT-6.1-Sol as MagicBrush teacher."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from openai import AzureOpenAI

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
PARQUET = ROOT / 'qwen_image_edit_smoke/data/public_raw/osunlp_MagicBrush/train-00000-of-00051-9fd9f23e2b1cb397.parquet'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_teacher_magicbrush import PROMPT, SYSTEM, metrics, run_code, save_bytes  # noqa: E402

AZURE_KEY = '1sofKlOHrz5LRE1K84Ix7HN45NIFeepv_GPT_AK'
AZURE_ENDPOINT = 'https://search.bytedance.net/gpt/openapi/online/multimodal/crawl'
TRAEX_URL = 'http://127.0.0.1:9584/v1/responses'
_AZURE = None


def azure_client():
    global _AZURE
    if _AZURE is None:
        _AZURE = AzureOpenAI(api_key=AZURE_KEY, api_version='2023-07-01-preview',
                             azure_endpoint=AZURE_ENDPOINT, timeout=240)
    return _AZURE


def traex_key():
    result = subprocess.run([sys.executable, str(ROOT / 'traex_api_proxy/get_local_key.py')],
                            capture_output=True, text=True, timeout=20, check=True)
    return result.stdout.strip()


def parse_json(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def call_gpt55(prompt, b64):
    response = azure_client().chat.completions.create(
        model='gpt-5.5-2026-04-24',
        messages=[{'role': 'system', 'content': SYSTEM},
                  {'role': 'user', 'content': [
                      {'type': 'text', 'text': prompt},
                      {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}],
        max_completion_tokens=8192, reasoning_effort='low')
    return (response.choices[0].message.content or '').strip()


def call_traex(model, prompt, b64, timeout, key):
    body = {
        'model': model,
        'input': [
            {'role': 'system', 'content': [{'type': 'input_text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [
                {'type': 'input_text', 'text': prompt},
                {'type': 'input_image', 'image_url': 'data:image/png;base64,' + b64}]},
        ],
        'max_output_tokens': 8192,
        'reasoning': {'effort': 'low'},
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


def save_resized(data, path, mode='RGB', size=512):
    image = Image.open(io.BytesIO(data)).convert(mode)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def process_row(row, out, backend, key):
    sample_id = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
    source = out / 'images' / f'{sample_id}_source.png'
    target = out / 'images' / f'{sample_id}_target.png'
    mask = out / 'images' / f'{sample_id}_mask.png'
    save_resized(row['source_img']['bytes'], source, 'RGB')
    save_resized(row['target_img']['bytes'], target, 'RGB')
    save_resized(row['mask_img']['bytes'], mask, 'L')
    b64 = base64.b64encode(source.read_bytes()).decode()
    prompt = PROMPT.format(instruction=row['instruction'])
    t0 = time.time()
    if backend == 'gpt55':
        text = call_gpt55(prompt, b64)
    elif backend == 'deepseek':
        text = call_traex('traex/DeepSeek-V4.1-Flash', prompt, b64, 180, key)
    else:
        text = call_traex('traex/GPT-6.1-Sol', prompt, b64, 900, key)
    elapsed = round(time.time() - t0, 1)
    (out / 'raw' / f'{sample_id}.txt').write_text(text)
    parsed = parse_json(text)
    execution = run_code(parsed.get('executable_code', ''), source,
                         out / 'executions' / f'{sample_id}.png')
    m = metrics(source, target, out / 'executions' / f'{sample_id}.png', mask) \
        if execution['status'] == 'passed' else {}
    return {
        'id': sample_id, 'task': 'rgb_to_editing_code',
        'source_image': str(source), 'target_image': str(target), 'mask_image': str(mask),
        'instruction': row['instruction'], 'teacher_backend': backend,
        'teacher_elapsed_seconds': elapsed,
        'reasoning': {k: parsed.get(k) for k in
                      ['visual_evidence', 'target_object', 'target_region',
                       'editing_intent', 'editing_plan', 'preservation_constraints']},
        'code': parsed.get('executable_code', ''),
        'requires_inpaint': parsed.get('requires_inpaint', False),
        'execution': execution, 'metrics': m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['gpt55', 'deepseek', 'gpt61'])
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / 'qwen_pixel' / out
    out.mkdir(parents=True, exist_ok=True)
    for sub in ['raw', 'images', 'executions']:
        (out / sub).mkdir(exist_ok=True)
    if args.fresh or not (out / 'samples.jsonl').exists():
        (out / 'samples.jsonl').write_text('')
    done = {json.loads(line)['id'] for line in (out / 'samples.jsonl').read_text().splitlines() if line.strip()}
    table = pq.read_table(PARQUET)
    n = min(args.limit, table.num_rows)
    indices = [int(i * table.num_rows / n) for i in range(n)]
    if args.max_samples:
        indices = indices[:args.max_samples]
    key = traex_key() if args.backend != 'gpt55' else None
    rows = []
    for i in indices:
        row = {k: table[k][i].as_py() for k in table.column_names}
        sample_id = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
        if sample_id not in done:
            rows.append(row)
    print(f'pending={len(rows)} workers={args.workers}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, row, out, args.backend, key): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            sample_id = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
            try:
                sample = future.result()
            except Exception as exc:
                print(json.dumps({'id': sample_id, 'error': type(exc).__name__,
                                  'message': str(exc)[:400]}), flush=True)
                continue
            with (out / 'samples.jsonl').open('a') as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(json.dumps({'id': sample_id, 'elapsed': sample['teacher_elapsed_seconds'],
                              'requires_inpaint': sample['requires_inpaint'],
                              'execution': sample['execution']['status'],
                              'metrics': sample['metrics']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
