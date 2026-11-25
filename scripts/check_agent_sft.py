"""Verify real multimodal prefixes and assistant-only labels before training."""
import json
import sys
from pathlib import Path

from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from trajectory_io import encode_trajectory


def main():
    processor = AutoProcessor.from_pretrained(sys.argv[1])
    processor.image_processor.size = {'shortest_edge': 512 * 512, 'longest_edge': 512 * 512}
    records = []
    for line in (ROOT / 'data/agent_demo_v1/train.jsonl').open():
        row = json.loads(line)
        batch = encode_trajectory(processor, row, 'cpu', True, 8192)
        if batch is None:
            raise ValueError('Trajectory exceeds the smoke context limit.')
        labels = batch['labels'][0]
        supervised = processor.tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=False)
        if '<tool_call>' not in supervised:
            raise ValueError('Expected a supervised tool call.')
        record = {'id': row['id'], 'sequence_tokens': int(labels.numel()),
                  'supervised_tokens': int((labels != -100).sum()),
                  'context_tokens_masked': int((labels == -100).sum()),
                  'image_count': int(batch['image_grid_thw'].shape[0]),
                  'target_action': row['target_message']['tool_calls'][0]['function']['name']}
        print(json.dumps(record), flush=True)
        records.append(record)
    (ROOT / 'data/agent_demo_v1/encoding_check.json').write_text(json.dumps(records, indent=2))
    print('TRAJECTORY_ENCODING_PASSED', flush=True)


if __name__ == '__main__':
    main()
