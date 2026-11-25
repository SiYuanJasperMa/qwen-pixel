"""Compare Qwen3.5-9B Instruct and GPT-5.5 on 10 COCO grounding samples."""
import argparse
import base64
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from openai import AzureOpenAI

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
sys.path.insert(0, str(ROOT / 'qwen_pixel/train'))
from run_pixel_sft import encode, load  # noqa: E402

REFCOCO = ROOT / 'qwen_image_edit_smoke/data/refcoco_train_real_scale_v1/train.jsonl'
SYSTEM = ('You are Qwen Pixel, a visual understanding model. For GROUND tasks output only JSON: '
          '{"bbox": [x1,y1,x2,y2]}. Do not add explanations.')
GPT_PROMPT = '''GROUND: Locate the object described as: {expr}
Return only JSON: {{"bbox": [x1,y1,x2,y2]}}'''


def parse_bbox(text):
    match = re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
    if not match:
        return None
    return [float(v) for v in match.groups()]


def iou(a, b):
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expression(instruction):
    marker = 'object described as:'
    return instruction.split(marker, 1)[1].strip() if marker in instruction else instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--model', default=str(ROOT / 'robojudge_v4q/models/Qwen3.5-9B'))
    parser.add_argument('--output', default=str(ROOT / 'qwen_pixel/outputs/compare_coco_grounding.jsonl'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in REFCOCO.read_text().splitlines() if line.strip()]
    seen = set()
    samples = []
    for row in rows:
        scene = str(row.get('scene_id') or row['id'])
        if scene in seen:
            continue
        seen.add(scene)
        samples.append(row)
        if len(samples) >= args.limit:
            break
    model, processor = load(SimpleNamespace(model=args.model, adapter=None, mode='generate'))
    model.eval()
    eos = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
           processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    client = AzureOpenAI(api_key=__import__("os").environ["QWEN_PIXEL_PROXY_KEY"],
                         api_version='2023-07-01-preview',
                         azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
                         timeout=240)
    results = []
    for row in samples:
        expr = expression(row['instruction'])
        image_path = Path(row['image'])
        if not image_path.is_absolute():
            image_path = ROOT / 'qwen_image_edit_smoke' / image_path
        target = [float(v) for v in json.loads(row['bbox'])] if isinstance(row['bbox'], str) else row['bbox']
        prompt = f'GROUND: Locate the object described as: {expr}'
        inputs = encode(processor, {'system': SYSTEM, 'instruction': prompt, 'image': str(image_path)},
                        model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True,
                                 eos_token_id=eos,
                                 pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
        qwen_text = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        response = client.chat.completions.create(
            model='gpt-5.5-2026-04-24',
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': GPT_PROMPT.format(expr=expr)},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}],
            max_completion_tokens=8192, reasoning_effort='low')
        gpt_text = (response.choices[0].message.content or '').strip()
        qwen_bbox = parse_bbox(qwen_text)
        gpt_bbox = parse_bbox(gpt_text)
        result = {
            'id': row['id'], 'expression': expr, 'target': target,
            'qwen_text': qwen_text, 'qwen_bbox': qwen_bbox, 'qwen_iou': iou(qwen_bbox, target),
            'gpt_text': gpt_text, 'gpt_bbox': gpt_bbox, 'gpt_iou': iou(gpt_bbox, target),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    summary = {
        'samples': len(results),
        'qwen_success': sum(r['qwen_iou'] >= 0.5 for r in results) / len(results),
        'gpt55_success': sum(r['gpt_iou'] >= 0.5 for r in results) / len(results),
        'qwen_mean_iou': sum(r['qwen_iou'] for r in results) / len(results),
        'gpt55_mean_iou': sum(r['gpt_iou'] for r in results) / len(results),
    }
    (out.parent / 'compare_coco_grounding_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
