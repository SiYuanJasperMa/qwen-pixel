"""Online GRPO smoke for one-shot pixel programs, with a frozen SFT reference.

Every completion is sampled once, executed once and judged on the actual image.
No repair conversation, agent loop, fabricated reward or teacher-forced answer.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pixel.oneshot_io import encode_oneshot, VERSION, image_path
from pixel.teacher_api import judge_edit, make_client
from run_pixel_sft import load
from build_teacher_magicbrush_inpaint import PROMPT, SYSTEM, parse_json, run_code
from build_magicbrush_target_informed import looks_hardcoded
from pixel.decoding import configure_decoding, decoding_identity


def reference_digest(model):
    digest=hashlib.sha256()
    names=[]
    for name,parameter in sorted(model.named_parameters()):
        if '.reference.'not in name:continue
        names.append(name);digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if not names:raise RuntimeError('Missing initial SFT reference adapter.')
    return digest.hexdigest()


def logps(model, inputs, temperature=1.):
    labels = inputs['labels'][0, 1:]
    mask = labels != -100
    result = model(**{k: v for k, v in inputs.items() if k != 'labels'}, use_cache=False)
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    chunks = []
    def chunk_logps(logits, targets):
        return -F.cross_entropy(logits.float() / temperature, targets, reduction='none')
    for pos in positions.split(128):
        logits, targets = result.logits[0, pos], labels[pos]
        if torch.is_grad_enabled():
            chunks.append(checkpoint(chunk_logps, logits, targets, use_reentrant=False))
        else:
            chunks.append(chunk_logps(logits, targets))
    return torch.cat(chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-groups', type=int, default=8)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--max-seq-len', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--beta', type=float, default=.01)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--decode-policy', choices=['legacy','deterministic'], default='legacy')
    args = p.parse_args()
    if args.group_size < 2 or args.updates < 1 or args.temperature <= 0:
        raise ValueError('Need at least two completions per group and one update.')
    random.seed(args.seed); torch.manual_seed(args.seed)
    configure_decoding(args.decode_policy)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with Path(args.tasks).open() as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    if not tasks or any(t.get('split') != 'train' for t in tasks):
        raise ValueError('RL smoke only accepts explicitly training-split tasks.')
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='train', image_max_pixels=262144))
    print('REFERENCE_INITIALIZING', flush=True)
    model.add_adapter('reference', copy.deepcopy(model.peft_config['default']))
    parameters_by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in parameters_by_name.items():
            if '.reference.' in name:
                original = parameters_by_name[name.replace('.reference.', '.default.')]
                parameter.data = original.detach().clone()
    model.set_adapter('default')
    initial_reference_sha256=reference_digest(model)
    print('REFERENCE_READY frozen initial SFT weights', flush=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    client = make_client()
    records, updates = [], 0
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    for group in range(args.max_groups):
        task = tasks[group % len(tasks)]
        row = {'id': task['id'], 'image': task['image'], 'system': SYSTEM,
               'instruction': PROMPT.format(instruction=task['instruction'])}
        prompt = encode_oneshot(processor, row, model.device)
        n = prompt['input_ids'].shape[1]
        samples = []
        model.eval()
        for j in range(args.group_size):
            folder = out / f'group_{group:02d}' / f'sample_{j:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            print('GENERATING', group, j, task['id'], flush=True)
            with torch.no_grad():
                ids = model.generate(**prompt, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=1., top_k=0,
                    repetition_penalty=1., use_cache=True, eos_token_id=eos,
                    pad_token_id=processor.tokenizer.pad_token_id)[0, n:].tolist()
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
            (folder / 'completion.txt').write_text(raw)
            print('GENERATED', group, j, len(ids), flush=True)
            record = {'task_id': task['id'], 'sample': j, 'sampled_tokens': len(ids),
                      'serializer_version': VERSION, 'truncated': not ids or ids[-1] != eos}
            execution, judgment, code = {}, None, ''
            try:
                if record['truncated']:
                    execution={'status':'truncated','output':None,'error':'No sampled assistant EOS; do not execute partial outputs.'}
                    raise ValueError(execution['error'])
                parsed = parse_json(raw)
                code = parsed.get('executable_code', '')
                blocked,reason=looks_hardcoded(code)
                execution = {'status':'rejected','output':None,'error':reason} if blocked else run_code(code, image_path(row), folder / 'edited.png')
                if execution['status'] == 'passed':
                    judgment = judge_edit(client, image_path(row), execution['output'],
                                          task['instruction'],strict_extent=True)
            except Exception as exc:
                record['error'] = type(exc).__name__ + ': ' + str(exc)[:250]
            reward = 0.
            pending = execution.get('status') == 'passed' and judgment is None
            if judgment is not None:
                # Pass has highest reward; a semantically eligible partial edit
                # can receive a smaller shaped outcome reward. Syntax alone: 0.
                reward = 1. if judgment['pass'] else .5 * judgment['reward']
            record.update(execution=execution, judge=judgment, reward=reward,
                          judge_pending=pending, code_sha256=hashlib.sha256(code.encode()).hexdigest())
            sample_row = {**row, 'target_token_ids': ids}
            inputs = encode_oneshot(processor, sample_row, model.device, True, args.max_seq_len) if ids else None
            if inputs is None:
                pending = True; record['score_error'] = 'context_limit'
            old = reference = None
            if not pending and inputs is not None:
                with torch.no_grad():
                    old = logps(model, inputs, args.temperature).cpu()
                    model.set_adapter('reference')
                    reference = logps(model, inputs, args.temperature).cpu()
                    model.set_adapter('default')
            record['scoring_complete']=old is not None and reference is not None
            (folder / 'outcome.json').write_text(json.dumps(record, indent=2))
            samples.append({'row': sample_row, 'old': old, 'reference': reference,
                            'reward': reward, 'pending': pending})
            print('ROLLOUT', json.dumps(record), flush=True)
        rewards = torch.tensor([s['reward'] for s in samples])
        record = {'group': group, 'task_id': task['id'], 'rewards': rewards.tolist(),
                  'reward_std': float(rewards.std(unbiased=False))}
        if any(s['pending'] or s['old'] is None for s in samples):
            record.update(updated=False, reason='incomplete_scoring_or_judge; retry separately')
        elif record['reward_std'] < 1e-6:
            record.update(updated=False, reason='equal_rewards; no fabricated advantages')
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
            record['advantages'] = advantages.tolist()
            model.train(); model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            probes = params[:8]
            before = [v.detach().float().cpu().clone() for v in probes]
            loss_sum = 0.
            for sample, advantage in zip(samples, advantages):
                inputs = encode_oneshot(processor, sample['row'], model.device, True, args.max_seq_len)
                current = logps(model, inputs, args.temperature)
                old = sample['old'].to(current.device)
                ref = sample['reference'].to(current.device)
                ratio = torch.exp(current - old)
                surrogate = torch.minimum(ratio * advantage.to(current.device),
                    ratio.clamp(.8, 1.2) * advantage.to(current.device))
                delta = (ref - current).clamp(-20, 20)
                kl = torch.exp(delta) - delta - 1
                loss = (-surrogate + args.beta * kl).mean() / len(samples)
                loss.backward(); loss_sum += float(loss.detach())
                del inputs, current, loss
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Invalid RL gradient.')
            optimizer.step()
            if reference_digest(model)!=initial_reference_sha256:
                raise RuntimeError('The initial SFT reference adapter changed during RL.')
            change = sum(float((v.detach().float().cpu() - b).abs().sum())
                         for v, b in zip(probes, before))
            if change <= 0:
                raise RuntimeError('No measured adapter update.')
            updates += 1
            record.update(updated=True, loss=loss_sum, grad_norm=float(norm),
                          parameter_delta_l1=change)
        records.append(record)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print('GRPO_GROUP', json.dumps(record), flush=True)
        if updates >= args.updates:
            break
    model.set_adapter('default')
    model.save_pretrained(out / 'final_adapter', selected_adapters=['default'])
    processor.save_pretrained(out / 'final_adapter')
    summary = {'algorithm': 'oneshot_online_GRPO', 'reference': 'frozen initial SFT adapter',
        'initial_reference_sha256':initial_reference_sha256,
        'final_reference_sha256':reference_digest(model),'judge_rubric':'full_visible_target_v1',
        'serializer_version': VERSION, 'groups': len(records), 'optimizer_updates': updates,
        'numeric_settings':decoding_identity(args.decode_policy),
        'nonzero_update_verified': updates > 0, 'completed_requested_updates': updates >= args.updates,
        'groups_detail': records, 'configuration': vars(args)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('ONESHOT_GRPO_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
