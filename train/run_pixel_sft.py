"""Assistant-token LoRA SFT for Qwen Pixel.

Supports native multi-image agent trajectory prefixes and legacy single-turn
records. Current image-edit experiments supervise the next assistant action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pixel.oneshot_io import encode_oneshot, VERSION as ONESHOT_VERSION
DEFAULT_MODEL = PROJECT.parent / 'robojudge_v4q/models/Qwen3.8-27B'


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


def encode(processor, row, device, supervised=False, max_seq_len=None, nonzero_weight=1.0,geometry_weight=1.0):
    if 'messages' in row:
        from trajectory_io import encode_trajectory
        result=encode_trajectory(processor,row,device,supervised,max_seq_len)
        if result is not None and supervised and geometry_weight!=1.:
            from geometry_loss import attach_geometry_weights
            result=attach_geometry_weights(processor,result,geometry_weight)
        return result
    result = encode_oneshot(processor, row, device, supervised, max_seq_len)
    if result is None:
        return None
    if supervised:
        labels = result['labels']
        if nonzero_weight != 1.0:
            weights = torch.zeros_like(result['input_ids'], dtype=torch.float)
            tokenizer = processor.tokenizer
            for i, token_id in enumerate(result['input_ids'][0].tolist()):
                if labels[0, i].item() == -100:
                    continue
                token = tokenizer.decode([token_id])
                weights[0, i] = nonzero_weight if token in set('1234567') else 1.0
            result['token_weights'] = weights
    if max_seq_len and result['input_ids'].shape[1] > max_seq_len:
        return None
    return {k: v.to(device) for k, v in result.items()}


def load(args):
    from pixel.adapter_identity import validate_adapter_base
    identity=validate_adapter_base(args.adapter,args.model)
    if identity:print('ADAPTER_BASE_CONFIRMED',json.dumps(identity),flush=True)
    processor_path = args.adapter if args.adapter and (Path(args.adapter) / 'processor_config.json').exists() else args.model
    print('LOAD_PROCESSOR', processor_path, flush=True)
    processor = AutoProcessor.from_pretrained(processor_path)
    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
        pixels = getattr(args, 'image_max_pixels', 256 * 256)
        processor.image_processor.size = {'shortest_edge': pixels, 'longest_edge': pixels}
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
    if stats.get('agent_contract'):
        (dest / 'agent_contract.json').write_text(json.dumps(stats['agent_contract'], indent=2))
    (dest / 'run_state.json').write_text(json.dumps({
        'step': step, 'arguments': vars(args), 'stats': stats,
        'serializer': stats.get('data_format', 'legacy single-turn Qwen role/vision tokens'),
    }, indent=2))


def train(args):
    from continuation_state import read_state,restore,save_latest,file_hash
    continuation=read_state(args.resume_training_state)if getattr(args,'resume_training_state',None)else None
    if continuation:
        if args.adapter and Path(args.adapter).resolve()!=Path(continuation['checkpoint']).resolve():
            raise ValueError('Resume state belongs to a different adapter.')
        args.adapter=continuation['checkpoint']
    existing_log=Path(args.output)/'metrics.jsonl'
    if existing_log.exists()and existing_log.stat().st_size and not continuation:
        raise ValueError('Training output already has metrics; use a fresh output or exact resume state.')
    native_validate=None
    if getattr(args,'agent_val_tasks',None):
        # Bind this implementation before the long run, rather than mixing a
        # newly edited validator with already-loaded decoder modules at step450.
        from agent_validation import evaluate as native_validate
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if getattr(args, 'agent_val_tasks', None):
        from pixel.decoding import configure_decoding
        configure_decoding(args.agent_val_decode_policy)
    model, processor = load(args)
    from lora_scope import target_pattern
    target=target_pattern(getattr(args,'lora_scope','hybrid'))
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
             'lora_target_pattern':target,
             'initialized_from_adapter':args.adapter,
             'optimizer_state_restored':False,
             'training_records': len(rows), 'assistant_only_loss': True,
             'data_format': 'native Qwen multimodal/tool template; next assistant action'
                            if any('messages' in r for r in rows) else ONESHOT_VERSION}
    if getattr(args,'agent_val_tasks',None):
        from pixel.decoding import decoding_identity
        stats['training_numeric_settings']=decoding_identity(args.agent_val_decode_policy)
    from pixel.sft_contract import training_contract
    if any('messages'in r for r in rows):
        from trajectory_io import LABEL_POLICY
        stats['label_policy']=LABEL_POLICY
    contract = training_contract(rows, args.image_max_pixels)
    if contract:
        stats['agent_contract'] = contract
    identity={'model':str(Path(args.model).resolve()),'data_sha256':file_hash(args.data),
        'row_ids':[r['id']for r in rows],'label_policy':stats.get('label_policy',ONESHOT_VERSION),
        'torch_version':str(torch.__version__),'deterministic':torch.are_deterministic_algorithms_enabled(),
        'tf32':torch.backends.cuda.matmul.allow_tf32,
        'settings':{key:getattr(args,key,None)for key in ('steps','accum','rank','lr','warmup','seed','lora_scope',
            'geometry_weight','nonzero_weight','image_max_pixels','max_seq_len','supervised_token_budget','agent_val_decode_policy')}}
    param_names=[name for name,p in model.named_parameters()if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    index = list(range(len(rows)))
    random.shuffle(index)
    cursor = 0
    supervised_tokens = 0
    first_step=1;past_elapsed=0.
    if continuation:
        if identity!=continuation['identity']:raise ValueError('Training identity changed on resume.')
        step,index,cursor,supervised_tokens,past_elapsed=restore(continuation,optimizer,param_names)
        first_step=step+1;stats['optimizer_state_restored']=True
        stats['resumed_from_training_state']=str(args.resume_training_state)
        if first_step>args.steps:raise ValueError('Requested training was already completed.')
    start = time.time()
    for step in range(first_step, args.steps + 1):
        losses = []
        actions = []
        sample_ids = []
        for _ in range(args.accum):
            attempts = 0
            while True:
                if not index:
                    raise ValueError('SFT training set is empty.')
                if attempts >= len(index):
                    raise ValueError('All sampled SFT examples exceed the context limit.')
                attempts += 1
                if cursor == len(index):
                    random.shuffle(index)
                    cursor = 0
                row = rows[index[cursor]]
                cursor += 1
                inputs = encode(processor, row, model.device, supervised=True,
                                max_seq_len=args.max_seq_len, nonzero_weight=args.nonzero_weight,
                                geometry_weight=args.geometry_weight)
                if inputs is not None:
                    break
            weights = inputs.pop('token_weights', None)
            supervised_tokens += int((inputs['labels'] != -100).sum())
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
            target_message = row.get('target_message', {})
            calls = target_message.get('tool_calls', [])
            actions.append(calls[0]['function']['name'] if calls else row.get('task','text'))
            sample_ids.append(row['id'])
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm) or (step == first_step and norm <= 0):
            raise RuntimeError('Invalid or zero gradients')
        warm = min(1., step / max(args.warmup, 1))
        budget = getattr(args, 'supervised_token_budget', 0)
        progress = min(1., supervised_tokens / budget) if budget else step / max(args.steps, 1)
        cosine = .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * warm * cosine
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {'step': step, 'loss': sum(losses) / len(losses), 'sample_losses':losses,
                  'supervised_tokens': supervised_tokens,
                  'sample_actions':actions,'grad_norm': float(norm),
                  'sample_ids':sample_ids,
                  'elapsed_seconds':past_elapsed+time.time()-start,'lr': optimizer.param_groups[0]['lr'],
                  'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
        log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if step == first_step:
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
        budget_reached = bool(budget and supervised_tokens >= budget)
        stop_requested=getattr(args,'stop_after_step',0)==step
        if step % args.save_every == 0 or step == args.steps or budget_reached or stop_requested:
            checkpoint=dest/f'checkpoint-{step}'
            save(model,processor,checkpoint,args,step,stats)
            state_path=checkpoint/'training_state.pt'if getattr(args,'keep_checkpoint_training_state',False)else dest/'latest_training_state.pt'
            save_latest(state_path,checkpoint,optimizer,identity,param_names,
                        step,index,cursor,supervised_tokens,record['elapsed_seconds'])
            if args.checkpoint_hook:
                hook_log=(checkpoint/'evaluation_hook.log').open('w')
                process=subprocess.Popen([sys.executable,args.checkpoint_hook,'--checkpoint',str(checkpoint),
                    '--step',str(step)],cwd=PROJECT,stdin=subprocess.DEVNULL,
                    stdout=hook_log,stderr=subprocess.STDOUT,start_new_session=True)
                hook_log.close()
                print('CHECKPOINT_EVALUATION_STARTED',step,process.pid,flush=True)
        agent_every = getattr(args, 'agent_val_every', 0)
        if getattr(args, 'agent_val_tasks', None) and (step % agent_every == 0 or step == args.steps or budget_reached or stop_requested):
            checkpoint = dest / f'checkpoint-{step}'
            if not (checkpoint/'run_state.json').exists():
                save(model,processor,checkpoint,args,step,stats)
                state_path=checkpoint/'training_state.pt'if getattr(args,'keep_checkpoint_training_state',False)else dest/'latest_training_state.pt'
                save_latest(state_path,checkpoint,optimizer,identity,param_names,
                            step,index,cursor,supervised_tokens,record['elapsed_seconds'])
            record['agent_validation'] = native_validate(model, processor, checkpoint, args)
            log.write(json.dumps(record)+'\n')
            print('TRAINING_RESUMES_AFTER_AGENT_VALIDATION',step,flush=True)
        if budget_reached:
            print('SUPERVISED_TOKEN_BUDGET_REACHED', supervised_tokens, budget, flush=True)
            break
        if stop_requested:
            print('TRAINING_CHECKPOINT_STOP',step,'planned_horizon',args.steps,flush=True)
            break
    log.close()
    print('TRAINING_COMPLETE'if step==args.steps or budget_reached else'TRAINING_CHECKPOINT_SAVED',dest/f'checkpoint-{step}',flush=True)


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
    parser.add_argument('--image-max-pixels', type=int, default=256 * 256)
    parser.add_argument('--lora-scope', choices=['hybrid','standard','attention'], default='hybrid')
    parser.add_argument('--geometry-weight',type=float,default=1.)
    parser.add_argument('--checkpoint-hook',default=None)
    parser.add_argument('--resume-training-state',default=None)
    parser.add_argument('--keep-checkpoint-training-state',action='store_true')
    parser.add_argument('--stop-after-step',type=int,default=0)
    parser.add_argument('--supervised-token-budget', type=int, default=0,
                        help='Stop after this assistant-token budget; steps remains a safety cap.')
    parser.add_argument('--agent-val-tasks')
    parser.add_argument('--agent-val-every',type=int,default=400)
    parser.add_argument('--agent-val-split',default='val')
    parser.add_argument('--agent-val-limit',type=int,default=0)
    parser.add_argument('--agent-val-profile',choices=['rgb','clean_edit'],default='clean_edit')
    parser.add_argument('--agent-val-decode-policy',choices=['legacy','deterministic','fp32_state'],default='deterministic')
    args = parser.parse_args()
    if args.agent_val_tasks and args.agent_val_every <= 0:
        parser.error('--agent-val-every must be positive for native editing validation.')
    train(args) if args.mode == 'train' else generate(args)


if __name__ == '__main__':
    main()
