"""Real native rollout regression for exact sampled-token SFT exports."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'train'))
from run_pixel_sft import load, encode
from run_agent_demo import rollout
from pixel.decoding import configure_decoding, DecodePolicyModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Choose a fresh regression output.')
    args.out.mkdir(parents=True)
    task = json.loads((ROOT/'data/sft_exact_replay_v1/tasks.jsonl').read_text().splitlines()[0])
    adapter = ROOT/'outputs/agent_sft_eos_label_regression_20261125/train/checkpoint-20'
    model_path = ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B'
    configure_decoding('deterministic')
    actor, processor = load(SimpleNamespace(model=str(model_path), adapter=str(adapter),
                            mode='generate', image_max_pixels=262144))
    summary, sampled = rollout(DecodePolicyModel(actor, 'deterministic'), processor,
        task['image'], task['instruction'], args.out/'actor', max_steps=8,
        max_new_tokens=4096, temperature=0., model_name=str(model_path), adapter=str(adapter),
        system_profile='rgb', use_checkpoint_interface=True)
    if not summary['complete']:
        raise ValueError('Native regression rollout did not complete.')
    saved = [json.loads(l) for l in (args.out/'actor/sft_steps.jsonl').read_text().splitlines()]
    if len(saved) != len(sampled) or len({r['id'] for r in saved}) != len(saved):
        raise ValueError('Exported action count or identity mismatch.')
    stop = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    for row, original in zip(saved, sampled):
        if row['target_token_ids'] != original['target_token_ids']:
            raise ValueError('Export lost the original sampled token sequence.')
        prompt = encode(processor, row, 'cpu')
        full = encode(processor, row, 'cpu', supervised=True)
        n = prompt['input_ids'].shape[1]
        target = torch.tensor(row['target_token_ids'], dtype=full['input_ids'].dtype)
        if not torch.equal(full['input_ids'][0, n:], target) or target[-1].item() != stop:
            raise ValueError('SFT inputs differ from sampled native action through EOS.')
        if (full['labels'][0, :n] != -100).any() or not torch.equal(full['labels'][0, n:], target):
            raise ValueError('Native assistant-only supervision mismatch.')
    # The evaluator reference is read only after the actor has finished.
    reference = adapter/'agent_validation'/task['id']/'edited_rgb.npy'
    if not reference.exists():
        raise ValueError('Missing frozen regression RGB reference.')
    expected = np.load(reference, allow_pickle=False)
    actual = np.load(args.out/'actor/edited_rgb.npy', allow_pickle=False)
    exact = np.array_equal(actual, expected)
    report = {'complete': True, 'actions': summary['actions'], 'exported_actions': len(saved),
              'exact_sampled_token_ids': True, 'assistant_only_eos_labels': True,
              'exact_frozen_reference_rgb': exact, 'adapter': str(adapter),
              'scope': 'single existing training photo regression; not unseen editing success'}
    (args.out/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not exact:
        raise ValueError('New rollout did not reproduce the frozen RGB reference.')


if __name__ == '__main__':
    main()
