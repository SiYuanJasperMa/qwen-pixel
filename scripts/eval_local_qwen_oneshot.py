"""One-shot teacher-style baseline for a local Qwen model on MagicBrush dev tasks.

The model sees the source image + the teacher prompt and must return the JSON with
`executable_code`. The code is executed and judged; no agent tool loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_pixel_sft import load  # noqa: E402
from build_teacher_multi import SYSTEM  # noqa: E402
from build_teacher_magicbrush_inpaint import PROMPT, metrics, parse_json, run_code  # noqa: E402
from build_magicbrush_oneshot_code_sft import SYSTEM as CODE_SYSTEM, INSTRUCTION as CODE_INSTRUCTION  # noqa: E402
from pixel.teacher_api import judge_edit, make_client  # noqa: E402
from pixel.oneshot_io import encode_oneshot, VERSION as ONESHOT_VERSION
from build_magicbrush_target_informed import looks_hardcoded
from pixel.decoding import configure_decoding, DecodePolicyModel, decoding_identity


def strip_fences(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--tasks', default=str(ROOT / 'data/magicbrush_dev8_v1/tasks.jsonl'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--prompt-mode', choices=['json', 'code'], default='json')
    parser.add_argument('--decode-policy', choices=['legacy','deterministic','fp32_state'], default='legacy')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'executions').mkdir(exist_ok=True)
    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    tasks = tasks[:args.limit]
    configure_decoding(args.decode_policy)
    run_identity = {'model':args.model,'adapter':args.adapter,'decode_policy':args.decode_policy,
        'tasks_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        'task_ids':[t['id']for t in tasks],'max_new_tokens':args.max_new_tokens,
        'prompt_mode':args.prompt_mode,'serializer_version':ONESHOT_VERSION,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.adapter:
        run_identity['adapter_sha256']=hashlib.sha256((Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()
    identity_path=out/'eval_run.json'
    if identity_path.exists()and json.loads(identity_path.read_text())!=run_identity:
        raise ValueError('Evaluation output has a different configuration; choose a fresh output.')
    identity_path.write_text(json.dumps(run_identity,indent=2))
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter, mode='generate',
                                            image_max_pixels=512 * 512))
    model=DecodePolicyModel(model,args.decode_policy)
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    client = make_client()
    results = []
    for task in tasks:
        sid = task['id']
        source = Path(task['image'])
        image = Image.open(source).convert('RGB')
        if args.prompt_mode == 'code':
            prompt = CODE_INSTRUCTION.format(instruction=task['instruction'])
            system_text = CODE_SYSTEM
        else:
            prompt = PROMPT.format(instruction=task['instruction'])
            system_text = SYSTEM
        inputs = encode_oneshot(processor, {'id': sid, 'image': str(source),
            'system': system_text, 'instruction': prompt}, model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, eos_token_id=eos,
                                       pad_token_id=processor.tokenizer.pad_token_id)
        tokens = generated[0, inputs['input_ids'].shape[1]:]
        raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        (out / f'{sid}.txt').write_text(raw)
        record = {'id': sid, 'instruction': task['instruction'], 'raw_chars': len(raw),
                  'serializer_version': ONESHOT_VERSION,
                  'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(tokens.numel()),
                  'truncated':not tokens.numel()or int(tokens[-1])!=eos}
        try:
            if record['truncated']:
                raise ValueError('Incomplete assistant output; do not execute truncated code.')
            if args.prompt_mode == 'code':
                match = re.search(r'(def edit\(image\):.*)', raw, re.S)
                code = match.group(1) if match else strip_fences(raw)
                record['parsed'] = True
            else:
                parsed = parse_json(strip_fences(raw))
                code = parsed.get('executable_code', '') or ''
                record['parsed'] = True
            record['code_chars'] = len(code)
            blocked,reason=looks_hardcoded(code)
            execution = {'status':'rejected','output':None,'error':reason} if blocked else run_code(code, source, out / 'executions' / f'{sid}.png')
            record['execution'] = execution
            if execution['status'] == 'passed':
                record['judge'] = judge_edit(client, str(source), execution['output'],
                                             task['instruction'], strict_extent=True)
        except Exception as exc:
            record.setdefault('parsed', False)
            record['judge_pending'] = (record.get('parsed') is True and
                                      (record.get('execution') or {}).get('status') == 'passed')
            record['error'] = f'{type(exc).__name__}: {exc}'[:400]
        results.append(record)
        print(json.dumps({'id': sid, 'parsed': record.get('parsed'),
                          'exec': (record.get('execution') or {}).get('status'),
                          'pass': (record.get('judge') or {}).get('pass')}), flush=True)
    (out / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    executed = sum(1 for r in results if (r.get('execution') or {}).get('status') == 'passed')
    passed = sum(1 for r in results if (r.get('judge') or {}).get('pass'))
    pending = sum(bool(r.get('judge_pending')) for r in results)
    summary = {'tasks': len(results), 'parsed': sum(1 for r in results if r.get('parsed')),
               'judge_rubric':'full_visible_target_v1',
               'decode_policy':args.decode_policy,'numeric_settings':decoding_identity(args.decode_policy),
               'executed': executed, 'judge_pass': passed, 'judge_pending': pending,
               'overall_pass_rate': (passed / len(results) if results else 0) if not pending else None,
               'executed_pass_rate': (passed / executed if executed else 0) if not pending else None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
