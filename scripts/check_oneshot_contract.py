"""Exercise real Qwen image preprocessing, loss masks and sampled-token replay."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from pixel.oneshot_io import encode_oneshot, VERSION
from run_pixel_sft import encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size = {'shortest_edge': 262144, 'longest_edge': 262144}
    paths = ['sft_all_v14', 'sft_ultraedit_oneshot_all_v1', 'sft_sciforma_oneshot_all_v1']
    results = []
    for name in paths:
        with (ROOT / 'data' / name / 'train.jsonl').open() as f:
            row = json.loads(next(f))
        train_prompt = encode(processor, row, 'cpu')
        eval_prompt = encode_oneshot(processor, row, 'cpu')
        full = encode(processor, row, 'cpu', True, nonzero_weight=2.)
        n = train_prompt['input_ids'].shape[1]
        assert torch.equal(train_prompt['input_ids'], eval_prompt['input_ids'])
        assert torch.equal(full['input_ids'][:, :n], eval_prompt['input_ids'])
        assert (full['labels'][:, :n] == -100).all()
        assert (full['labels'][:, n:] != -100).all()
        assert (full['token_weights'][:, :n] == 0).all()
        sampled = full['input_ids'][0, n:].tolist()
        replay = encode_oneshot(processor, {**row, 'target_token_ids': sampled}, 'cpu', True)
        assert torch.equal(replay['input_ids'], full['input_ids'])
        assert torch.equal(replay['pixel_values'], full['pixel_values'])
        assert torch.equal(replay['labels'], full['labels'])
        for key, value in full.items():
            if key == 'token_weights':
                continue  # SFT weighting is intentionally absent from RL policy scoring.
            if isinstance(value, torch.Tensor) and value.shape == full['input_ids'].shape:
                assert torch.equal(replay[key], value), 'RL auxiliary tensor mismatch: ' + key
        results.append({'dataset': name, 'id': row['id'], 'prompt_tokens': n,
            'assistant_tokens': len(sampled), 'prompt_parity': True,
            'assistant_only_mask': True, 'exact_rl_token_replay': True})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'version': VERSION, 'checks': results}, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
