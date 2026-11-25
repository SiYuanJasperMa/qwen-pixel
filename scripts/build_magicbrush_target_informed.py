"""STaR-style target-informed teacher distillation for MagicBrush.

The teacher sees SOURCE + TARGET (+ optional changed-region mask) and writes code
that reproduces the target. The student never sees the target at inference; we
guard against leakage with a hardcoded-code filter and a flip-based reusability
probe.
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageChops, ImageStat

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_magicbrush_inpaint import parse_json, run_code, PROMPT as RFT_PROMPT  # noqa: E402
from build_teacher_multi import SYSTEM, azure_client, traex_key  # noqa: E402
from pixel.teacher_api import image_block, judge_edit, make_client  # noqa: E402
from pixel.data_lineage import canonical_task, source_group, split_of_group
from pixel.source_policy import source_disabled,dataset_of

TRAEX_URL = 'http://127.0.0.1:9584/v1/responses'
GPT62_URL = 'http://127.0.0.1:9588/v1/responses'
GPT62_KEY_FILE = ROOT.parent / 'traex_api_proxy/proxy_logs_gpt62/local.key'
TEACHER_MODELS = {
    'gpt62': 'traex/GPT-6.2-Sol',
    'gpt61': 'traex/GPT-6.1-Sol',
    'terra': 'traex/GPT-5.6-Terra',
    'luna': 'traex/GPT-5.6-Luna',
    'seed21pro': 'traex/Seed-2.1-Pro-0915',
    'seed21turbo': 'traex/Seed-2.1-Turbo',
    'deepseek': 'traex/DeepSeek-V4.1-Flash',
    'gpt56sol': 'traex/GPT-5.6-Sol',
}

PROMPT = '''TASK: RGB_TO_EDITING_CODE (TARGET-INFORMED)

Image 1 is the SOURCE image. Image 2 is the TARGET image (the desired result).
{extra_images}

INSTRUCTION:
{instruction}

Analyze what changed between SOURCE and TARGET, then write complete, runnable
Python code that edits the SOURCE image to be as close as possible to TARGET.

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

Hard requirements for target-informed data:
- The code is executed later with ONLY the source image. It must not read the
  target image, the mask, or any external file.
- Use reusable semantic operations (color/threshold segmentation, masking,
  inpainting, compositing, texture transfer). Do NOT hardcode target pixel
  values or large literal arrays. Coordinates derived from the source image are allowed.
- Preserve everything outside the intended edit region.
- The student will see only SOURCE and INSTRUCTION. Ground visual_evidence in
  SOURCE and phrase plans as actions executable from SOURCE + INSTRUCTION.
  Do not write claims that the student observes TARGET, a target mask, or an
  unavailable reference. TARGET is teacher-only construction evidence.

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

CRITICAL OUTPUT FORMAT:
Your executable_code field MUST define exactly:
    def edit(image):
        ...
        return image
It will be imported and called as edit(source_image). Do NOT use argparse,
sys.argv, __main__, input(), or any command-line interface.
'''


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def call_multi(teacher, prompt, images, key, effort='high', return_metadata=False):
    if teacher == 'gpt55':
        content = [{'type': 'text', 'text': prompt}] + [image_block(str(p)) for p in images]
        resp = azure_client().chat.completions.create(
            model='gpt-5.5-2026-04-24',
            messages=[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user', 'content': content}],
            max_completion_tokens=8192, reasoning_effort='low')
        text = (resp.choices[0].message.content or '').strip()
        meta = {'model': resp.model, 'usage': resp.usage.model_dump() if resp.usage else None}
        return (text, meta) if return_metadata else text
    model = TEACHER_MODELS[teacher]
    content = [{'type': 'input_text', 'text': prompt}]
    content += [{'type': 'input_image', 'image_url': 'data:image/png;base64,' + b64(p)} for p in images]
    body = {
        'model': model,
        'input': [{'role': 'system', 'content': [{'type': 'input_text', 'text': SYSTEM}]},
                  {'role': 'user', 'content': content}],
        'max_output_tokens': 8192,
        'reasoning': {'effort': effort},
    }
    if teacher == 'gpt62':
        with requests.Session() as session:
            session.trust_env = False
            resp = session.post(GPT62_URL, json=body, timeout=(5, 1800),
                                headers={'Authorization': 'Bearer ' + GPT62_KEY_FILE.read_text().strip()})
    else:
        resp = requests.post(TRAEX_URL, json=body, timeout=1800,
                             headers={'Authorization': 'Bearer ' + key})
    resp.raise_for_status()
    data = resp.json()
    if teacher == 'gpt62' and (data.get('status') != 'completed' or data.get('model') != model):
        raise RuntimeError(f'GPT-6.2-Sol response not complete or model mismatch: {data.get("status")}')
    texts = []
    for message in data.get('output', []):
        for part in message.get('content', []):
            if part.get('type') in ('output_text', 'text'):
                texts.append(part.get('text', ''))
    text = '\n'.join(texts).strip()
    if not text:
        raise RuntimeError('Teacher has no final visible output; status=' + str(data.get('status')))
    meta = {'model': data.get('model', model), 'usage': data.get('usage')}
    return (text, meta) if return_metadata else text


PROGRAM_FILTER_VERSION = 'source_only_program_v2'


def looks_hardcoded(code):
    # Program length and local variable names do not establish target leakage.
    # In particular mask_image=Image.new(...) and source-derived coordinates
    # are normal image editing operations.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, 'invalid_python'
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(n.split('.')[0] in ('os', 'subprocess', 'requests', 'socket', 'urllib',
                                       'base64', 'pickle') for n in names):
                return True, 'external_or_encoded_dependency'
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                    'open', 'load_image', 'exec', 'eval', '__import__'):
                return True, 'external_read_or_dynamic_execution'
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ('open', 'imread', 'read_bytes', 'read_text', 'memmap'):
                    return True, 'external_read'
                if attr in ('load', 'loadtxt', 'genfromtxt') and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id in ('np', 'numpy'):
                    return True, 'external_read'
    if re.search(r'["\'][^"\']*\.(?:png|jpe?g|webp)["\']', code, re.I):
        return True, 'file_reference'
    for m in re.finditer(r'\[([^\[\]]{2000,})\]', code):
        if len(re.findall(r'-?\d+(?:\.\d+)?', m.group(1))) > 300:
            return True, 'large_array_literal'
    if re.search(r'(\d+(?:\.\d+)?\s*,\s*){60,}', code):
        return True, 'long_numeric_literal'
    return False, ''


def changed_ratio(a, b):
    diff = ImageChops.difference(a.convert('RGB'), b.convert('RGB'))
    stat = ImageStat.Stat(diff)
    return sum(stat.mean) / (3 * 255.0)


def process(sample, out, key, client, teacher, use_mask, mode='ti', strict_extent=False,
            keep_output=False,reuse_raw=False):
    if source_disabled(sample):
        return {'id':sample['id'],'task_id':sample.get('task_id',sample['id']),
                'dataset':dataset_of(sample),'teacher':teacher,'generation_mode':mode,
                'instruction':sample.get('instruction',''),'status':'source_disabled'}
    source = Path(sample['source_image'])
    target = Path(sample['target_image'])
    images = [source, target] if mode == 'ti' else [source]
    extra = ''
    if mode == 'ti' and use_mask and sample.get('mask_image') and Path(sample['mask_image']).exists():
        images.append(Path(sample['mask_image']))
        extra = 'Image 3 is the changed-region mask (white = edited region).'
    prompt = (PROMPT.format(instruction=sample['instruction'], extra_images=extra)
              if mode == 'ti' else RFT_PROMPT.format(instruction=sample['instruction']))
    with Image.open(source) as source_image:
        width,height=source_image.size
    prompt=prompt.replace('The source image is 512x512.',f'The source image is {width}x{height}.')
    prompt += '\nWrite all supervision explanations and plans in English.\n'
    t0 = time.time()
    record = {'id': sample['id'], 'teacher': teacher, 'target_informed': mode == 'ti',
              'instruction': sample['instruction'], 'source_image': str(source),
              'target_image': str(target), 'mask_image': sample.get('mask_image'),
              'use_mask': use_mask}
    task_id = sample.get('task_id', canonical_task(sample['id']))
    group = sample.get('source_group_id') or source_group(task_id, source)
    record.update(task_id=task_id, dataset=sample.get('dataset', 'magicbrush'),
                  source_group_id=group, split=split_of_group(group),
                  generation_mode=mode, rft_failure_observed=sample.get('rft_failure_observed'),
                  rft_success_observed=sample.get('rft_success_observed'))
    try:
        raw_path = out/'raw'/f"{sample['id']}.txt"
        meta_path = out/'raw'/f"{sample['id']}.meta.json"
        if reuse_raw and raw_path.is_file():
            text = raw_path.read_text()
            api_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'model':None,'usage':None}
            record['teacher_response_reused_from_raw'] = True
            record['raw_response_metadata_missing'] = not meta_path.exists()
        else:
            text, api_meta = call_multi(teacher, prompt, images, key, return_metadata=True)
            raw_path.write_text(text)
            meta_path.write_text(json.dumps(api_meta))
        record.update(teacher_model=api_meta.get('model'), teacher_usage=api_meta.get('usage'))
        parsed = parse_json(text)
        code = parsed.get('executable_code', '') or ''
        hardcoded, reason = looks_hardcoded(code)
        exec_path = Path('/tmp/qwen_pixel_ti_exec') / f"{teacher}_{uuid.uuid4().hex}.png"
        exec_path.parent.mkdir(parents=True, exist_ok=True)
        execution = ({'status': 'rejected', 'error': reason, 'output': None}
                     if hardcoded else run_code(code, source, exec_path))
    except json.JSONDecodeError as exc:
        record.update(status='format_failed', error=f'Invalid visible JSON: {exc}'[:300],
                      elapsed_seconds=round(time.time()-t0,1))
        return record
    except Exception as exc:
        record.update({'status': 'teacher_error', 'error': f'{type(exc).__name__}: {exc}'[:400],
                       'elapsed_seconds': round(time.time() - t0, 1)})
        return record
    record.update({'status': execution['status'], 'code': code, 'execution': execution,
                   'hardcoded': hardcoded, 'hardcoded_reason': reason,
                   'program_filter_version': PROGRAM_FILTER_VERSION,
                   'requires_inpaint': parsed.get('requires_inpaint', False),
                   'elapsed_seconds': round(time.time() - t0, 1),
                   'reasoning': {k: parsed.get(k) for k in
                                 ['visual_evidence', 'target_object', 'target_region',
                                  'editing_intent', 'editing_plan', 'preservation_constraints']}})
    if execution['status'] == 'passed':
        try:
            record['judge'] = judge_edit(client, str(source), execution['output'],
                                         sample['instruction'], strict_extent=strict_extent)
        except Exception as exc:
            record['judge'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        if (record.get('judge') or {}).get('pass'):
            record['status'] = 'pass'
        # flip probe: same object, mirrored layout. Color/segmentation code should
        # still run; hardcoded coordinates usually break or become no-ops.
        try:
            flip = out / 'reuse' / f"{sample['id']}_flip.png"
            flip.parent.mkdir(parents=True, exist_ok=True)
            Image.open(source).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT).save(flip)
            reuse = run_code(code, flip, out / 'reuse' / f"{sample['id']}_out.png")
            record['reuse_exec'] = reuse['status']
            if reuse['status'] == 'passed':
                record['reuse_changed'] = changed_ratio(Image.open(flip), Image.open(reuse['output']))
            for tmp in (flip, Path(reuse['output']) if reuse.get('output') else None):
                try:
                    if tmp:
                        tmp.unlink()
                except Exception:
                    pass
        except Exception as exc:
            record['reuse_exec'] = 'error'
            record['reuse_error'] = f'{type(exc).__name__}: {exc}'[:200]
    if keep_output and execution.get('output') and Path(execution['output']).is_file():
        from pixel.rgb_text_eval import pixel_metrics
        try:
            record['pixel_metrics'] = pixel_metrics(source, target, execution['output'])
        except ValueError as exc:
            record.update(status='failed', pixel_metrics_error=str(exc))
            record['execution']['status']='invalid_output_dimensions'
            record['judge']={'pass':False,'reason':'Output dimensions violate the aligned RGB contract.'}
        if (record.get('judge') or {}).get('pass') or (record.get('judge') or {}).get('error'):
            destination = out / 'executions' / f"{sample['id']}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(execution['output']).read_bytes())
            record['retained_output'] = str(destination)
    try:
        if execution.get('output'):
            Path(execution['output']).unlink()
    except Exception:
        pass
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accepted', default=str(ROOT / 'data/teacher_magicbrush_merged_v1/judged.jsonl'))
    parser.add_argument('--tasks', help='Shared source-group task manifest; bypass accepted-only pool')
    parser.add_argument('--rescue-only', action='store_true')
    parser.add_argument('--attempt-index', type=int, default=0)
    parser.add_argument('--mode', choices=['rft', 'ti'], default='ti')
    parser.add_argument('--teacher', required=True,
                        choices=['gpt55', *TEACHER_MODELS])
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    out = Path(args.out or ROOT / 'data' / f'teacher_target_informed_{args.teacher}')
    for sub in ['raw', 'executions', 'reuse']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.tasks or args.accepted).read_text().splitlines() if line.strip()]
    if args.tasks:
        if any(r.get('split') != 'train' for r in rows):
            raise ValueError('Teacher synthesis expects training tasks only.')
        if args.rescue_only:
            rows = [r for r in rows if r.get('ti_rescue_eligible')]
        rows = [{**r, 'task_id': r['id'],
                 'id': f"{r['id']}__{args.mode}_{args.teacher}_{args.attempt_index}",
                 'source_image': r.get('source_image', r['image'])} for r in rows]
    else:
        if args.rescue_only:
            raise ValueError('--rescue-only requires the full task manifest via --tasks')
        rows = [r for r in rows if r.get('judge') and r['judge'].get('pass')]
    judged_path = out / 'judged.jsonl'
    done = set()
    if judged_path.exists():
        done = {json.loads(line)['id'] for line in judged_path.read_text().splitlines()
                if line.strip() and json.loads(line).get('status') != 'teacher_error'}
    rows = [r for r in rows if r['id'] not in done and not source_disabled(r)]
    if args.limit:
        rows = rows[:args.limit]
    print(json.dumps({'teacher': args.teacher, 'pending': len(rows), 'done': len(done),
                      'use_mask': args.use_mask}), flush=True)
    key = None if args.teacher in ('gpt55', 'gpt62') else traex_key()
    client = make_client()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r, out, key, client, args.teacher, args.use_mask, args.mode): r
                   for r in rows}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                print(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:300]}), flush=True)
                continue
            with judged_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'id': record['id'], 'exec': record['status'],
                              'pass': (record.get('judge') or {}).get('pass'),
                              'hardcoded': record.get('hardcoded'),
                              'reuse': record.get('reuse_exec')}), flush=True)
    all_rows = [json.loads(line) for line in judged_path.read_text().splitlines() if line.strip()]
    executed = sum(1 for r in all_rows if r['status'] in ('pass', 'passed', 'judge_fail'))
    passed = sum(1 for r in all_rows if (r.get('judge') or {}).get('pass'))
    summary = {'teacher': args.teacher, 'use_mask': args.use_mask, 'tasks': len(all_rows),
               'executed': executed, 'judge_pass': passed,
               'overall_pass_rate': passed / len(all_rows) if all_rows else 0,
               'hardcoded': sum(1 for r in all_rows if r.get('hardcoded')),
               'reuse_passed': sum(1 for r in all_rows if r.get('reuse_exec') == 'passed')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
