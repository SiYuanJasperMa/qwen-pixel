"""Assistant-token LoRA SFT for Qwen Pixel.

Supports text-only palette tasks (T2P, PU) and image-conditioned tasks (I2P,
EDIT, GROUND, QA). The target palette/grid is never placed in the prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT.parent / 'qwen_image_edit_smoke/models/Qwen3.5-9B-Base'


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def prefix(row):
    head = '<|im_start|>system\n' + row['system'] + '<|im_end|>\n<|im_start|>user\n'
    if row.get('image'):
        head += '<|vision_start|><|image_pad|><|vision_end|>\n'
    return head + row['instruction'] + '<|im_end|>\n<|im_start|>assistant\n'


def resolve_image(row):
    path = Path(row['image'])
    return path if path.is_absolute() else PROJECT / path


def encode(processor, row, device, supervised=False, max_seq_len=None, nonzero_weight=1.0):
    image = Image.open(resolve_image(row)).convert('RGB') if row.get('image') else None
    ptext = prefix(row)
    kwargs = {'text': [ptext], 'return_tensors': 'pt'}
    if image is not None:
        kwargs['images'] = [image]
    prompt = processor(**kwargs)
    if supervised:
        kwargs['text'] = [ptext + row['answer'] + '<|im_end|>']
        result = processor(**kwargs)
        n = prompt.input_ids.shape[1]
        if not torch.equal(result.input_ids[:, :n], prompt.input_ids):
            raise ValueError(f'Prompt/answer boundary mismatch in {row["id"]}')
        labels = result.input_ids.clone()
        labels[:, :n] = -100
        if not (labels != -100).any():
            raise ValueError(f'No assistant tokens in {row["id"]}')
        result['labels'] = labels
        if nonzero_weight != 1.0:
            weights = torch.zeros_like(result.input_ids, dtype=torch.float)
            tokenizer = processor.tokenizer
            for i, token_id in enumerate(result.input_ids[0].tolist()):
                if labels[0, i].item() == -100:
                    continue
                token = tokenizer.decode([token_id])
                weights[0, i] = nonzero_weight if token in set('1234567') else 1.0
            result['token_weights'] = weights
    else:
        result = prompt
    if max_seq_len and result['input_ids'].shape[1] > max_seq_len:
        return None
    return {k: v.to(device) for k, v in result.items()}


def load(args):
    print('LOAD_PROCESSOR', args.model, flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
        processor.image_processor.size = {'shortest_edge': 256 * 256, 'longest_edge': 256 * 256}
    print('LOAD_MODEL', args.model, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': int(os.environ.get('LOCAL_RANK', '0'))})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=args.mode == 'train')
    print('MODEL_READY', str(model.device), flush=True)
    return model, processor


def save(model, processor, dest, args, step, stats):
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    (dest / 'run_state.json').write_text(json.dumps({
        'step': step, 'arguments': vars(args), 'stats': stats,
        'serializer': 'explicit Qwen role and vision tokens; palette/grid text',
    }, indent=2))


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, processor = load(args)
    target = (r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|'
              r'in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj)')
    config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
                        target_modules=target, bias='none', task_type='CAUSAL_LM')
    if not args.adapter:
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    model.print_trainable_parameters()
    rows = read_rows(args.data)
    if args.tasks:
        allowed = {t.strip() for t in args.tasks.split(',') if t.strip()}
        rows = [r for r in rows if r['task'] in allowed]
    if args.limit:
        rows = rows[:args.limit]
    val = read_rows(args.val) if args.val else []
    if args.tasks:
        val = [r for r in val if r['task'] in allowed]
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    log = (dest / 'metrics.jsonl').open('a', buffering=1)
    initial = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
               if p.requires_grad and 'lora_B' in n}
    stats = {'trainable_parameters': sum(p.numel() for p in params),
             'training_records': len(rows), 'assistant_only_loss': True}
    optimizer.zero_grad(set_to_none=True)
    index = list(range(len(rows)))
    random.shuffle(index)
    cursor = 0
    start = time.time()
    for step in range(1, args.steps + 1):
        losses = []
        for _ in range(args.accum):
            while True:
                if cursor == len(index):
                    random.shuffle(index)
                    cursor = 0
                row = rows[index[cursor]]
                cursor += 1
                inputs = encode(processor, row, model.device, supervised=True,
                                max_seq_len=args.max_seq_len, nonzero_weight=args.nonzero_weight)
                if inputs is not None:
                    break
            weights = inputs.pop('token_weights', None)
            result = model(**inputs, use_cache=False)
            if weights is not None:
                shift_logits = result.logits[:, :-1, :]
                shift_labels = inputs['labels'][:, 1:]
                shift_weights = weights[:, 1:]
                token_losses = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1), reduction='none', ignore_index=-100)
                mask = shift_labels.reshape(-1) != -100
                loss = (token_losses[mask] * shift_weights.reshape(-1)[mask]).sum() / \
                       shift_weights.reshape(-1)[mask].sum()
            else:
                loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss on {row["id"]}')
            (loss / args.accum).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm) or (step == 1 and norm <= 0):
            raise RuntimeError('Invalid or zero gradients')
        warm = min(1., step / max(args.warmup, 1))
        cosine = .1 + .9 * .5 * (1 + math.cos(math.pi * step / max(args.steps, 1)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * warm * cosine
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {'step': step, 'loss': sum(losses) / len(losses), 'grad_norm': float(norm),
                  'elapsed_seconds': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                  'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
        log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if step == 1:
            delta = sum(float((p.detach().float().cpu() - initial[n]).abs().sum())
                        for n, p in model.named_parameters() if n in initial)
            if not delta > 0:
                raise RuntimeError('Adapter weights did not change')
            stats.update(first_step_nonzero_grad=True, first_step_lora_B_delta_l1=delta)
            (dest / 'gradient_check.json').write_text(json.dumps(stats, indent=2))
            initial.clear()
        if args.eval_every and step % args.eval_every == 0 and val:
            model.eval()
            vl = []
            with torch.no_grad():
                for r in val[:args.eval_limit]:
                    inputs = encode(processor, r, model.device, supervised=True,
                                    max_seq_len=args.max_seq_len)
                    if inputs is not None:
                        vl.append(float(model(**inputs, use_cache=False).loss))
            record['val_loss'] = sum(vl) / len(vl) if vl else None
            log.write(json.dumps(record) + '\n')
            model.train()
        if step % args.save_every == 0 or step == args.steps:
            save(model, processor, dest / f'checkpoint-{step}', args, step, stats)
    log.close()
    print('TRAINING_COMPLETE', dest / f'checkpoint-{args.steps}', flush=True)


def generate(args):
    model, processor = load(args)
    model.eval()
    rows = read_rows(args.data)
    if args.tasks:
        allowed = {t.strip() for t in args.tasks.split(',') if t.strip()}
        rows = [r for r in rows if r['task'] in allowed]
    if args.limit:
        rows = rows[:args.limit]
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {r['id'] for r in read_rows(dest)} if dest.exists() else set()
    eos = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
           processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    with dest.open('a', buffering=1) as out:
        for row in rows:
            if row['id'] in done:
                continue
            inputs = encode(processor, row, model.device)
            start = time.time()
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     use_cache=True, eos_token_id=eos,
                                     pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            text = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            result = {'id': row['id'], 'task': row['task'], 'text': text,
                      'seconds': time.time() - start, 'model': str(args.model),
                      'adapter': args.adapter, 'max_new_tokens': args.max_new_tokens,
                      'generated_tokens': int(ids.shape[1] - inputs['input_ids'].shape[1])}
            out.write(json.dumps(result) + '\n')
            print(row['id'], round(result['seconds'], 2), repr(text[:160]), flush=True)
    print('GENERATION_COMPLETE', dest, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'generate'])
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--adapter')
    parser.add_argument('--data', required=True)
    parser.add_argument('--val')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-limit', type=int, default=8)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--tasks', default=None, help='comma-separated task filter')
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--nonzero-weight', type=float, default=5.0,
                        help='CE weight for grid tokens 1..7; 1.0 disables weighting')
    parser.add_argument('--max-new-tokens', type=int, default=1400)
    args = parser.parse_args()
    train(args) if args.mode == 'train' else generate(args)


if __name__ == '__main__':
    main()
