"""Generate GPT-5.5 teacher samples for MagicBrush train rows."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from openai import AzureOpenAI
import pyarrow.parquet as pq

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
PARQUET = ROOT / 'qwen_image_edit_smoke/data/public_raw/osunlp_MagicBrush/train-00000-of-00051-9fd9f23e2b1cb397.parquet'
OUT = ROOT / 'qwen_pixel/data/teacher_magicbrush_v1'

SYSTEM = (
    'You are a visual reasoning and image editing teacher. Inspect the attached source image '
    'and the instruction. Return one JSON object only. Do not use markdown fences. '
    'Be concrete and do not invent objects that are not visible.'
)

PROMPT = '''TASK: RGB_TO_EDITING_CODE

INSTRUCTION:
{instruction}

Return JSON with exactly these fields:
- visual_evidence: list of short visible facts that justify the edit
- target_object: the object or region to edit
- target_region: a short description of the target area
- editing_intent: what should change
- editing_plan: ordered list of steps
- preservation_constraints: list of things that must not change
- executable_code: a Python function `def edit(image):` that returns a PIL Image, or "" if the edit needs semantic generation
- requires_inpaint: true if the edit needs generating new content that PIL/NumPy cannot produce

Available helpers: load_image(path), save_image(image, path), crop(image, box), resize(image, size),
recolor(image, mask, color), composite(source, edited, mask), preserve(source, edited, mask).
You may also use PIL and NumPy directly. Do not use the network, shell, or external files.
The source image is 512x512. Use `w, h = image.size` and relative coordinates; never hardcode a fixed resolution.
'''


def extract_text(response):
    return (response.choices[0].message.content or '').strip()


def parse_json(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def save_bytes(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run_code(code, source_path, out_path):
    if not code.strip():
        return {'status': 'skipped', 'error': 'empty code', 'output': None}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input.png').write_bytes(source_path.read_bytes())
        script = tmp / 'run.py'
        script.write_text(
            'import numpy as np\n'
            'from PIL import Image, ImageOps, ImageDraw, ImageFilter\n'
            'def load_image(path): return Image.open(path).convert("RGB")\n'
            'def save_image(image, path): image.save(path)\n'
            'def crop(image, box): return image.crop(box)\n'
            'def resize(image, size): return image.resize(size, Image.Resampling.LANCZOS)\n'
            'def recolor(image, mask, color):\n'
            '    a=np.array(image).copy(); m=np.array(mask.convert("L"))>127; a[m]=color; return Image.fromarray(a)\n'
            'def composite(source, edited, mask): return Image.composite(edited, source, mask.convert("L"))\n'
            'def preserve(source, edited, mask): return composite(source, edited, mask)\n'
            + code + '\n'
            'if __name__ == "__main__":\n'
            '    img=load_image("input.png")\n'
            '    out=edit(img)\n'
            '    out.save("output.png")\n'
        )
        try:
            result = subprocess.run([sys.executable, str(script)], cwd=tmp, text=True,
                                    capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'timeout', 'output': None}
        if result.returncode != 0:
            return {'status': 'failed', 'error': (result.stderr or '')[-800:], 'output': None}
        if not (tmp / 'output.png').exists():
            return {'status': 'failed', 'error': 'no output.png', 'output': None}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes((tmp / 'output.png').read_bytes())
        return {'status': 'passed', 'error': '', 'output': str(out_path)}


def metrics(source_path, target_path, pred_path, mask_path):
    source = Image.open(source_path).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    target = Image.open(target_path).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    pred = Image.open(pred_path).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
    s, t, p, m = map(lambda im: np.asarray(im, dtype=np.float32) / 255.0,
                     [source, target, pred, mask])
    m = m > 0.5
    l1 = float(np.abs(p - t).mean())
    no_op = float(np.abs(s - t).mean())
    mask_l1 = float(np.abs(p - t)[m].mean()) if m.any() else None
    preserve = float(np.abs(p - t)[~m].mean()) if (~m).any() else None
    return {'l1': l1, 'no_op_l1': no_op, 'gain': no_op - l1,
            'mask_l1': mask_l1, 'preserve_l1': preserve}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--model', default='gpt-5.5-2026-04-24')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw').mkdir(exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    (out / 'executions').mkdir(exist_ok=True)

    table = pq.read_table(PARQUET)
    client = AzureOpenAI(
        api_key=__import__("os").environ["QWEN_PIXEL_PROXY_KEY"],
        api_version='2023-07-01-preview',
        azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
        timeout=240,
    )
    rows = []
    n = min(args.limit, table.num_rows)
    indices = [int(i * table.num_rows / n) for i in range(n)]
    for i in indices:
        row = {k: table[k][i].as_py() for k in table.column_names}
        sample_id = f"magicbrush_train_{row['img_id']}_{row['turn_index']}"
        source = out / 'images' / f'{sample_id}_source.png'
        target = out / 'images' / f'{sample_id}_target.png'
        mask = out / 'images' / f'{sample_id}_mask.png'
        save_bytes(row['source_img']['bytes'], source)
        save_bytes(row['target_img']['bytes'], target)
        save_bytes(row['mask_img']['bytes'], mask)
        b64 = base64.b64encode(source.read_bytes()).decode()
        content = [
            {'type': 'text', 'text': PROMPT.format(instruction=row['instruction'])},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
        ]
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{'role': 'system', 'content': SYSTEM},
                          {'role': 'user', 'content': content}],
                max_completion_tokens=8192,
            )
            text = extract_text(response)
            elapsed = round(time.time() - t0, 1)
            (out / 'raw' / f'{sample_id}.txt').write_text(text)
            parsed = parse_json(text)
            execution = run_code(parsed.get('executable_code', ''), source,
                                 out / 'executions' / f'{sample_id}.png')
            m = metrics(source, target, out / 'executions' / f'{sample_id}.png', mask) \
                if execution['status'] == 'passed' else {}
            sample = {
                'id': sample_id,
                'task': 'rgb_to_editing_code',
                'source_image': str(source),
                'target_image': str(target),
                'mask_image': str(mask),
                'instruction': row['instruction'],
                'teacher_model': args.model,
                'teacher_elapsed_seconds': elapsed,
                'reasoning': {k: parsed.get(k) for k in
                              ['visual_evidence', 'target_object', 'target_region',
                               'editing_intent', 'editing_plan', 'preservation_constraints']},
                'code': parsed.get('executable_code', ''),
                'requires_inpaint': parsed.get('requires_inpaint', False),
                'execution': execution,
                'metrics': m,
            }
            rows.append(sample)
            print(json.dumps({'id': sample_id, 'elapsed': elapsed,
                              'requires_inpaint': sample['requires_inpaint'],
                              'execution': execution['status'],
                              'metrics': m}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({'id': sample_id, 'error': type(exc).__name__,
                              'message': str(exc)[:500]}), flush=True)
    (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print('WROTE', out / 'samples.jsonl', len(rows), flush=True)


if __name__ == '__main__':
    main()
