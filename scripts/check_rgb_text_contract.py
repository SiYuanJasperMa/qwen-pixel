"""Actual Qwen tokenizer audit of long RGB SFT/inference prefixes."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.rgb_text_io import encode_text_oneshot


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parent / 'robojudge_v4q/models/Qwen3.8-27B')
    tasks = [json.loads(l) for l in (ROOT / 'data/rgb_text_smoke_v1/tasks.jsonl').read_text().splitlines()]
    results = []
    for task in tasks:
        answer = json.dumps({'executable_code': 'def edit(image):\n    return image.copy()'})
        prefix = encode_text_oneshot(tokenizer, task, 'cpu')
        full = encode_text_oneshot(tokenizer, task, 'cpu', answer=answer, max_seq_len=140000)
        n = prefix['input_ids'].shape[1]
        assert torch.equal(full['input_ids'][:, :n], prefix['input_ids'])
        assert torch.all(full['labels'][:, :n] == -100)
        assert torch.equal(full['labels'][:, n:], full['input_ids'][:, n:])
        assert set(prefix) == {'input_ids', 'attention_mask'}
        results.append({'id': task['id'], 'input_tokens': n,
                        'supervised_tokens': int((full['labels'] != -100).sum()),
                        'prefix_exact': True, 'pixels_truncated': False})
    out = ROOT / 'outputs/rgb_text_contract.json'
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
