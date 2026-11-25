"""Small online GRPO smoke over real Python/image-tool trajectories.

Each group contains independently sampled full agent rollouts for one task.
Only sampled assistant tokens are scored. The frozen base model is the KL
reference; this smoke uses one optimizer update per collected group.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pixel.teacher_api import judge_edit, make_client
from run_agent_demo import rollout
from run_pixel_sft import encode, load


def action_logps(model, inputs):
    labels = inputs['labels'][0, 1:]
    mask = labels != -100
    outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'}, use_cache=False)
    logits = outputs.logits[0, :-1][mask].float()
    return -F.cross_entropy(logits, labels[mask], reduction='none')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--max-new-tokens', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--temperature', type=float, default=.9)
    parser.add_argument('--beta', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--system-profile', choices=['default', 'rgb'], default='default')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='train', image_max_pixels=512 * 512))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    tasks = [json.loads(l) for l in Path(args.tasks).open()]
    tasks = [t for t in tasks if t.get('split', 'train') == 'train']
    if not tasks or args.group_size < 2:
        raise ValueError('Need train tasks and at least two sampled trajectories per group.')
    client = make_client()
    records = []
    updates = 0
    for group in range(args.groups):
        task = tasks[group % len(tasks)]
        samples = []
        model.eval()  # Keep probability scoring deterministic; LoRA dropout is disabled.
        for sample in range(args.group_size):
            workspace = out / f'group_{group:02d}' / f'sample_{sample:02d}'
            summary, rows = rollout(model, processor, task['image'], task['instruction'],
                workspace, args.max_steps, args.max_new_tokens, args.temperature,
                model_name=args.model, adapter=args.adapter, system_profile=args.system_profile)
            reward, judgment = 0., None
            if summary['complete']:
                judgment = judge_edit(client, workspace / 'source.png',
                    workspace / summary['validation']['image_path'], task['instruction'])
                reward = judgment['reward']
            (workspace / 'reward.json').write_text(json.dumps(
                {'reward': reward, 'judge': judgment, 'complete': summary['complete']}, indent=2))
            scored = []
            for row in rows:
                inputs = encode(processor, row, model.device, True, 8192)
                if inputs is None:
                    continue
                with torch.no_grad():
                    old = action_logps(model, inputs).cpu()
                    if args.beta:
                        with model.disable_adapter():
                            reference = action_logps(model, inputs).cpu()
                    else:
                        reference = old
                scored.append({'row': row, 'old': old, 'reference': reference})
            samples.append({'reward': reward, 'actions': scored, 'complete': summary['complete']})
            print('ROLLOUT_REWARD', group, sample, reward, len(scored), flush=True)
        rewards = torch.tensor([s['reward'] for s in samples], dtype=torch.float32)
        std = rewards.std(unbiased=False)
        record = {'group': group, 'task_id': task['id'], 'rewards': rewards.tolist(),
                  'reward_std': float(std), 'complete': sum(s['complete'] for s in samples)}
        if std < 1e-6:
            record.update(updated=False, reason='equal_group_rewards; no fabricated advantage')
        else:
            advantages = (rewards - rewards.mean()) / (std + 1e-4)
            record['advantages'] = advantages.tolist()
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            snapshot = parameters[0].detach().float().cpu().clone()
            total_loss = 0.
            for sample, advantage in zip(samples, advantages):
                token_count = sum(a['old'].numel() for a in sample['actions'])
                if not token_count:
                    continue
                for action in sample['actions']:
                    inputs = encode(processor, action['row'], model.device, True, 8192)
                    current = action_logps(model, inputs)
                    old = action['old'].to(current.device)
                    reference = action['reference'].to(current.device)
                    ratio = torch.exp(current - old)
                    clipped = ratio.clamp(.8, 1.2)
                    surrogate = torch.minimum(ratio * advantage.to(current.device),
                                              clipped * advantage.to(current.device))
                    delta = (reference - current).clamp(-20, 20)
                    kl = torch.exp(delta) - delta - 1
                    loss = (-surrogate + args.beta * kl).sum() / (token_count * len(samples))
                    loss.backward()
                    total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise RuntimeError('RL has invalid or zero gradients.')
            optimizer.step()
            delta = float((parameters[0].detach().float().cpu() - snapshot).abs().sum())
            if not delta > 0:
                raise RuntimeError('RL adapter parameters did not change.')
            updates += 1
            record.update(updated=True, loss=total_loss, grad_norm=float(norm),
                          parameter_delta_l1=delta)
            model.save_pretrained(out / f'checkpoint-group-{group + 1}')
            processor.save_pretrained(out / f'checkpoint-group-{group + 1}')
        records.append(record)
        (out / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        print('GRPO_GROUP', json.dumps(record), flush=True)
    summary = {'groups': len(records), 'optimizer_updates': updates,
               'nonzero_update_verified': updates > 0, 'groups_detail': records,
               'reference': 'frozen base Qwen without adapters',
               'algorithm': 'online grouped rollout GRPO; clipped token ratio, group-normalized outcome reward, optional KL; one update per group'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    model.save_pretrained(out / 'final_adapter')
    processor.save_pretrained(out / 'final_adapter')
    print('AGENT_GRPO_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
