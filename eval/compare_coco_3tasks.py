"""Compare Qwen3.5-9B Instruct and GPT-5.5 on 10 grounding + 10 caption + 10 QA COCO samples."""
import argparse
import base64
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from openai import AzureOpenAI

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
sys.path.insert(0, str(ROOT / 'qwen_pixel/train'))
from run_pixel_sft import encode, load  # noqa: E402

REFCOCO = ROOT / 'qwen_image_edit_smoke/data/refcoco_train_real_scale_v1/train.jsonl'
SYSTEM = ('You are Qwen Pixel, a visual understanding model. Answer directly and briefly. '
          'For GROUND output only JSON: {"bbox": [x1,y1,x2,y2]}. Do not add explanations.')


def parse_bbox(value):
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    return [float(v) for v in value]


def expression(instruction):
    marker = 'object described as:'
    return instruction.split(marker, 1)[1].strip() if marker in instruction else instruction


def parse_qwen_bbox(text):
    m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if m:
        vals = [float(v) for v in m.group(1).split(',')]
        return [v / 1000 * 256 for v in vals]
    m = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', text)
    if m:
        return [float(v) for v in m.group(1).split(',')]
    m = re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
    return [float(v) for v in m.groups()] if m else None


def parse_gpt_bbox(text):
    m = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', text)
    if m:
        return [float(v) for v in m.group(1).split(',')]
    m = re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
    return [float(v) for v in m.groups()] if m else None


def iou(a, b):
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = max(0, ax1 - ax0) * max(0, ay1 - ay0) + max(0, bx1 - bx0) * max(0, by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def short_answer(text):
    text = strip_think(text).lower()
    for word in ['left', 'right', 'yes', 'no']:
        if re.search(r'\b' + word + r'\b', text):
            return word
    m = re.search(r'\b(\d+)\b', text)
    return m.group(1) if m else text.strip()[:40]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(ROOT / 'robojudge_v4q/models/Qwen3.5-9B'))
    parser.add_argument('--output', default=str(ROOT / 'qwen_pixel/outputs/compare_coco_3tasks.jsonl'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in REFCOCO.read_text().splitlines() if line.strip()]
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[str(row.get('scene_id') or row['id'])].append(row)
    scenes = list(by_scene)[:10]
    ground = [by_scene[s][0] for s in scenes]
    qa = []
    for i, scene in enumerate(scenes):
        scene_rows = by_scene[scene]
        if i < 4:
            count = len({tuple(round(v, 2) for v in parse_bbox(r['bbox'])) for r in scene_rows})
            qa.append({'kind': 'count', 'row': scene_rows[0],
                       'instruction': 'QA: How many distinct annotated object regions are in this image?',
                       'answer': str(count)})
        elif i < 7:
            row = scene_rows[0]
            bbox = parse_bbox(row['bbox'])
            side = 'left' if (bbox[0] + bbox[2]) / 2 < 128 else 'right'
            qa.append({'kind': 'position', 'row': row,
                       'instruction': f'QA: Is the object described as "{expression(row["instruction"])}" '
                                      f'on the left or right half of the image? Answer left or right.',
                       'answer': side})
        else:
            row = scene_rows[0]
            if i == 7:
                expr, answer = expression(row['instruction']), 'yes'
            else:
                other = by_scene[scenes[(i + 3) % len(scenes)]][0]
                expr, answer = expression(other['instruction']), 'no'
            qa.append({'kind': 'presence', 'row': row,
                       'instruction': f'QA: Is there an object described as "{expr}" in this image? '
                                      f'Answer yes or no.', 'answer': answer})
    model, processor = load(SimpleNamespace(model=args.model, adapter=None, mode='generate'))
    model.eval()
    eos = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
           processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    client = AzureOpenAI(api_key=__import__("os").environ["QWEN_PIXEL_PROXY_KEY"],
                         api_version='2023-07-01-preview',
                         azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
                         timeout=240)

    def qwen_generate(image_path, instruction, max_new_tokens):
        inputs = encode(processor, {'system': SYSTEM, 'instruction': instruction, 'image': str(image_path)},
                        model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 eos_token_id=eos,
                                 pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
        return processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gpt_call(image_path, instruction):
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        response = client.chat.completions.create(
            model='gpt-5.5-2026-04-24',
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}],
            max_completion_tokens=8192, reasoning_effort='low')
        return (response.choices[0].message.content or '').strip()

    def image_of(row):
        path = Path(row['image'])
        return path if path.is_absolute() else ROOT / 'qwen_image_edit_smoke' / path

    results = []
    # Grounding
    for row in ground:
        image = image_of(row)
        instruction = f'GROUND: Locate the object described as: {expression(row["instruction"])}'
        target = parse_bbox(row['bbox'])
        qt = qwen_generate(image, instruction, 64)
        gt = gpt_call(image, instruction + '\nReturn only JSON: {"bbox": [x1,y1,x2,y2]}')
        qb, gb = parse_qwen_bbox(qt), parse_gpt_bbox(gt)
        results.append({'task': 'ground', 'id': row['id'], 'instruction': instruction,
                        'target': target, 'qwen_text': qt, 'gpt_text': gt,
                        'qwen_bbox': qb, 'gpt_bbox': gb,
                        'qwen_iou': iou(qb, target), 'gpt_iou': iou(gb, target)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    # Caption
    for row in ground:
        image = image_of(row)
        instruction = 'Describe this image in one natural sentence.'
        qt = qwen_generate(image, instruction, 96)
        gt = gpt_call(image, instruction)
        results.append({'task': 'caption', 'id': row['id'], 'instruction': instruction,
                        'qwen_text': qt, 'gpt_text': gt})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    # QA
    for item in qa:
        image = image_of(item['row'])
        qt = qwen_generate(image, item['instruction'], 32)
        gt = gpt_call(image, item['instruction'] + ' Answer with one word or number.')
        results.append({'task': 'qa', 'kind': item['kind'], 'id': item['row']['id'],
                        'instruction': item['instruction'], 'answer': item['answer'],
                        'qwen_text': qt, 'gpt_text': gt,
                        'qwen_answer': short_answer(qt), 'gpt_answer': short_answer(gt)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    ground_results = [r for r in results if r['task'] == 'ground']
    qa_results = [r for r in results if r['task'] == 'qa']
    summary = {
        'ground_qwen_success': sum(r['qwen_iou'] >= 0.5 for r in ground_results) / len(ground_results),
        'ground_gpt_success': sum(r['gpt_iou'] >= 0.5 for r in ground_results) / len(ground_results),
        'ground_qwen_iou': sum(r['qwen_iou'] for r in ground_results) / len(ground_results),
        'ground_gpt_iou': sum(r['gpt_iou'] for r in ground_results) / len(ground_results),
        'qa_qwen_exact': sum(r['qwen_answer'] == r['answer'] for r in qa_results) / len(qa_results),
        'qa_gpt_exact': sum(r['gpt_answer'] == r['answer'] for r in qa_results) / len(qa_results),
    }
    (out.parent / 'compare_coco_3tasks_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
