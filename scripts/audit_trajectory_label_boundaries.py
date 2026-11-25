"""Audit actual native tensors: past actions/images masked, one complete target."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'train'))
from trajectory_io import encode_trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--processor', type=Path,
        default=ROOT/'outputs/agent_sft_en_27b_37photos_geometry4_exposure30_20261125/train/checkpoint-1600')
    args = p.parse_args()
    payload = args.data.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    processor = AutoProcessor.from_pretrained(args.processor)
    processor.image_processor.size = {'shortest_edge':262144, 'longest_edge':262144}
    stop = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    report_rows, actions = [], Counter()
    for index, row in enumerate(rows):
        if row['target_message']['role'] != 'assistant':
            raise ValueError('Non-assistant target: '+row['id'])
        prompt = encode_trajectory(processor, row, 'cpu')
        full = encode_trajectory(processor, row, 'cpu', supervised=True)
        n = prompt['input_ids'].shape[1]
        if full['input_ids'].shape[1] > 8192:
            raise ValueError('Context overflow: '+row['id'])
        for key, original in prompt.items():
            result = full[key]
            actual = result[:, :n] if original.shape == prompt['input_ids'].shape else result
            if not torch.equal(original, actual):
                raise ValueError('Prompt/image tensor changed: '+row['id']+'/'+key)
        labels = full['labels'][0]
        positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
        expected = torch.arange(n, n+positions.numel())
        if not positions.numel() or not torch.equal(positions, expected):
            raise ValueError('Historical or discontinuous target labels: '+row['id'])
        if not torch.equal(labels[positions], full['input_ids'][0, positions]):
            raise ValueError('Labels differ from input target tokens: '+row['id'])
        if full['input_ids'][0, positions[-1]].item() != stop:
            raise ValueError('Target does not end at native EOS: '+row['id'])
        target = row['target_message']
        calls = target.get('tool_calls', [])
        action = calls[0]['function']['name'] if calls else 'raw_native'
        actions[action] += 1
        report_rows.append({'id':row['id'], 'prompt_tokens':n,
            'target_tokens':positions.numel(), 'context_tokens':full['input_ids'].shape[1],
            'historical_assistant_actions':sum(m['role']=='assistant' for m in row['messages']),
            'all_history_and_images_masked':True, 'native_prefix_tensors_equal':True,
            'assistant_target_through_eos':True})
        if (index+1) % 100 == 0:
            print('LABEL_BOUNDARIES_CHECKED',index+1,len(rows),flush=True)
    dataset_hash = hashlib.sha256(payload).hexdigest()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != dataset_hash:
        raise ValueError('Training dataset changed during audit.')
    summary = {'rows':len(rows), 'dataset_sha256':dataset_hash,
        'historical_assistant_actions_masked':sum(r['historical_assistant_actions'] for r in report_rows),
        'actions':dict(actions), 'max_context_tokens':max(r['context_tokens'] for r in report_rows),
        'max_assistant_tokens':max(r['target_tokens'] for r in report_rows),
        'native_image_and_prompt_parity':True, 'all_history_masked':True,
        'all_assistant_targets_through_eos':True, 'dataset_unchanged':True,
        'scope':'native tensor/label audit; not code execution or semantic editing success'}
    args.out.write_text(json.dumps({'summary':summary, 'rows':report_rows},indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
