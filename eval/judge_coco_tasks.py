"""Judge COCO-task edits without a target image (source + prediction + instruction)."""
import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
PROMPT = '''You are evaluating an image editing result.
Image 1: source image.
Image 2: model prediction.
Instruction: {instruction}
Score 1-5 for instruction_adherence, preservation, visual_quality, and set pass true only if the edit is clearly correct and usable.
Return JSON only: {{"instruction_adherence":1,"preservation":1,"visual_quality":1,"pass":false,"reason":"..."}}
'''


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def judge_row(row, client):
    pred = row.get('execution', {}).get('output')
    if not pred or not Path(pred).exists():
        return {'id': row['id'], 'skipped': True, 'reason': 'no prediction'}
    content = [
        {'type': 'text', 'text': PROMPT.format(instruction=row['instruction'])},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(row['source_image'])}},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(pred)}},
    ]
    response = client.chat.completions.create(
        model='gpt-5.5-2026-04-24', messages=[{'role': 'user', 'content': content}],
        max_completion_tokens=8192, reasoning_effort='low')
    text = (response.choices[0].message.content or '').strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {'raw': text}
    return {'id': row['id'], **parsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    client = AzureOpenAI(api_key=__import__("os").environ["QWEN_PIXEL_PROXY_KEY"],
                         api_version='2023-07-01-preview',
                         azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
                         timeout=240)
    rows = []
    for path in args.samples:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    out_path = Path(args.output)
    done = {json.loads(line)['id'] for line in out_path.read_text().splitlines() if line.strip()} if out_path.exists() else set()
    pending = [row for row in rows if row['id'] not in done]
    print(f'pending={len(pending)} workers={args.workers}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_row, row, client): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(json.dumps({'id': row['id'], 'error': type(exc).__name__,
                                  'message': str(exc)[:300]}), flush=True)
                continue
            with out_path.open('a') as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps(result, ensure_ascii=False)[:400], flush=True)
    print('WROTE', out_path, flush=True)


if __name__ == '__main__':
    main()
