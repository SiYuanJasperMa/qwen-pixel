"""Evaluate the actual final checkpoint of an assistant-token-budget run."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--out', required=True)
    p.add_argument('--gpu', required=True)
    args = p.parse_args()
    train, out = Path(args.train), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        last = json.loads((train / 'metrics.jsonl').read_text().splitlines()[-1])
        checkpoint = train / f"checkpoint-{last['step']}"
        state = json.loads((checkpoint / 'run_state.json').read_text())
        budget = state['arguments'].get('supervised_token_budget', 0)
        if budget and last.get('supervised_tokens', 0) < budget:
            raise ValueError('Training stopped before the requested token budget.')
        if not (checkpoint / 'adapter_model.safetensors').is_file():
            raise ValueError('Adapter weights missing.')
    except Exception as exc:
        (out / 'status.json').write_text(json.dumps({'status': 'training_incomplete',
            'error': type(exc).__name__ + ': ' + str(exc)}))
        return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'PYTHONUNBUFFERED': '1'}
    summaries = []
    for i in range(4):
        dest = out / f'shard{i}'
        with (out / f'shard{i}.log').open('w') as f:
            subprocess.run([sys.executable, str(ROOT / 'scripts/eval_local_qwen_oneshot.py'),
                '--model', state['arguments']['model'], '--adapter', str(checkpoint),
                '--tasks', str(ROOT / f'data/magicbrush_official_dev_v1/tasks_{i}.jsonl'),
                '--out', str(dest), '--limit', '16', '--max-new-tokens', '4096'],
                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        summaries.append(json.loads((dest / 'summary.json').read_text()))
    summary = {k: sum(s.get(k, 0) for s in summaries)
               for k in ('tasks', 'parsed', 'executed', 'judge_pass', 'judge_pending')}
    summary['overall_pass_rate'] = (summary['judge_pass'] / summary['tasks']
                                   if not summary['judge_pending'] else None)
    summary.update(checkpoint=str(checkpoint), supervised_tokens=last.get('supervised_tokens'),
                   evaluation='custom VLM judge on 64 official MagicBrush dev tasks')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
