"""Fair 10-sample QA comparison plus caption judging for Qwen vs GPT-5.5."""
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
QA_SYSTEM = ('You are Qwen Pixel, a visual understanding model. Answer the question with a single '
             'word or number. Do not output JSON, code, or explanations.')
CAPTION_PROMPT = '''Image: an image. Caption: {caption}
Score this caption 1-5 for accuracy and completeness. Return JSON only:
{{"score": 1, "pass": false, "reason": "..."}}'''


def parse_bbox(value):
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    return [float(v) for v in value]


def expression(instruction):
    marker = 'object described as:'
    return instruction.split(marker, 1)[1].strip() if marker in instruction else instruction


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
    parser.add_argument('--captions', default=str(ROOT / 'qwen_pixel/outputs/compare_coco_3tasks.jsonl'))
    parser.add_argument('--output', default=str(ROOT / 'qwen_pixel/outputs/compare_coco_qa_caption.jsonl'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in REFCOCO.read_text().splitlines() if line.strip()]
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[str(row.get('scene_id') or row['id'])].append(row)
    scenes = list(by_scene)[:10]
    qa = []
    # 4 relative-position questions.
    rel = 0
    for scene in scenes:
        scene_rows = by_scene[scene]
        boxes = {}
        for row in scene_rows:
            boxes.setdefault(tuple(round(v, 2) for v in parse_bbox(row['bbox'])), row)
        if len(boxes) < 2:
            continue
        (b1, r1), (b2, r2) = list(boxes.items())[:2]
        e1, e2 = expression(r1['instruction']), expression(r2['instruction'])
        answer = 'yes' if (b1[0] + b1[2]) / 2 < (b2[0] + b2[2]) / 2 else 'no'
        qa.append({'kind': 'relative', 'row': r1,
                   'instruction': f'QA: Is the object described as "{e1}" left of the object described as "{e2}"? Answer yes or no.',
                   'answer': answer})
        rel += 1
        if rel >= 4:
            break
    # 3 left/right questions.
    for scene in scenes:
        row = by_scene[scene][0]
        bbox = parse_bbox(row['bbox'])
        side = 'left' if (bbox[0] + bbox[2]) / 2 < 128 else 'right'
        qa.append({'kind': 'position', 'row': row,
                   'instruction': f'QA: Is the object described as "{expression(row["instruction"])}" on the left or right half of the image? Answer left or right.',
                   'answer': side})
        if len([q for q in qa if q['kind'] == 'position']) >= 3:
            break
    # 3 presence questions: two true, one clearly false.
    for i, scene in enumerate(scenes[:2]):
        row = by_scene[scene][0]
        qa.append({'kind': 'presence', 'row': row,
                   'instruction': f'QA: Is there an object described as "{expression(row["instruction"])}" in this image? Answer yes or no.',
                   'answer': 'yes'})
    row = by_scene[scenes[2]][0]
    qa.append({'kind': 'presence', 'row': row,
               'instruction': 'QA: Is there a purple elephant in this image? Answer yes or no.',
               'answer': 'no'})
    qa = qa[:10]
    model, processor = load(SimpleNamespace(model=args.model, adapter=None, mode='generate'))
    model.eval()
    eos = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
           processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    client = AzureOpenAI(api_key=__import__("os").environ["QWEN_PIXEL_PROXY_KEY"],
                         api_version='2023-07-01-preview',
                         azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
                         timeout=240)

    def image_of(row):
        path = Path(row['image'])
        return path if path.is_absolute() else ROOT / 'qwen_image_edit_smoke' / path

    def qwen_generate(image, instruction):
        inputs = encode(processor, {'system': QA_SYSTEM, 'instruction': instruction, 'image': str(image)},
                        model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=24, do_sample=False, use_cache=True,
                                 eos_token_id=eos,
                                 pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
        return processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gpt_call(image, instruction):
        b64 = base64.b64encode(Path(image).read_bytes()).decode()
        response = client.chat.completions.create(
            model='gpt-5.5-2026-04-24',
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}],
            max_completion_tokens=8192, reasoning_effort='low')
        return (response.choices[0].message.content or '').strip()

    results = []
    for item in qa:
        image = image_of(item['row'])
        qt = qwen_generate(image, item['instruction'])
        gt = gpt_call(image, item['instruction'] + ' Answer with one word.')
        results.append({'task': 'qa', 'kind': item['kind'], 'id': item['row']['id'],
                        'instruction': item['instruction'], 'answer': item['answer'],
                        'qwen_text': qt, 'gpt_text': gt,
                        'qwen_answer': short_answer(qt), 'gpt_answer': short_answer(gt)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    # Judge captions from the previous run.
    caption_rows = [json.loads(line) for line in Path(args.captions).read_text().splitlines() if line.strip()]
    caption_rows = [r for r in caption_rows if r['task'] == 'caption']
    scores = {'qwen': [], 'gpt': []}
    for r in caption_rows:
        image = image_of({'image': next(x['image'] for x in rows if x['id'] == r['id'])})
        for key in ['qwen', 'gpt']:
            b64 = base64.b64encode(Path(image).read_bytes()).decode()
            response = client.chat.completions.create(
                model='gpt-5.5-2026-04-24',
                messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': CAPTION_PROMPT.format(caption=r[key + '_text'])},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}],
                max_completion_tokens=8192, reasoning_effort='low')
            text = (response.choices[0].message.content or '').strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            try:
                judge = json.loads(text)
            except Exception:
                judge = {'score': 0, 'pass': False, 'raw': text}
            scores[key].append(judge)
            print(json.dumps({'task': 'caption_judge', 'id': r['id'], 'model': key, **judge}, ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    summary = {
        'qa_samples': len(results),
        'qa_qwen_exact': sum(r['qwen_answer'] == r['answer'] for r in results) / len(results),
        'qa_gpt_exact': sum(r['gpt_answer'] == r['answer'] for r in results) / len(results),
        'caption_qwen_mean_score': sum(s.get('score', 0) for s in scores['qwen']) / max(len(scores['qwen']), 1),
        'caption_gpt_mean_score': sum(s.get('score', 0) for s in scores['gpt']) / max(len(scores['gpt']), 1),
        'caption_qwen_pass': sum(bool(s.get('pass')) for s in scores['qwen']) / max(len(scores['qwen']), 1),
        'caption_gpt_pass': sum(bool(s.get('pass')) for s in scores['gpt']) / max(len(scores['gpt']), 1),
    }
    (out.parent / 'compare_coco_qa_caption_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
