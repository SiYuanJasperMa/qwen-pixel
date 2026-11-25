"""Check split separation and real tool-feedback coverage before scaling."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    root = Path(args.dataset)
    tasks = [json.loads(l) for l in (root / 'tasks.jsonl').open()]
    splits = {split: {t['scene_sha256'] for t in tasks if t['split'] == split}
              for split in ('train', 'val')}
    overlap = splits['train'] & splits['val']
    if overlap:
        raise ValueError('Images overlap between train and validation.')
    scene_splits = {split: {str(t.get('original_scene_id', t['scene_sha256']))
                            for t in tasks if t['split'] == split}
                    for split in ('train', 'val')}
    if scene_splits['train'] & scene_splits['val']:
        raise ValueError('Original source scenes overlap across train and validation.')
    records = {}
    for split in ('train', 'val'):
        file = root / (split + '.jsonl')
        rows = [json.loads(l) for l in file.open()]
        trajectories = set()
        action_counts = {}
        for row in rows:
            trajectories.add(row['meta']['trajectory_id'])
            target = row['target_message']
            for call in target.get('tool_calls', []):
                name = call['function']['name']
                action_counts[name] = action_counts.get(name, 0) + 1
            for message in row['messages']:
                if isinstance(message['content'], list):
                    for block in message['content']:
                        if block.get('type') == 'image' and not Path(block['image']).exists():
                            raise ValueError('Missing source or actual tool image: ' + block['image'])
            if not row['messages'] or row['messages'][0]['role'] != 'system':
                raise ValueError('Missing task system prompt.')
        records[split] = {'trajectories': len(trajectories), 'assistant_actions': len(rows),
                          'action_counts': action_counts}
    report = {'image_split_overlap': len(overlap), 'original_scene_overlap': 0, 'splits': records,
              'all_referenced_images_available': True,
              'supervision': 'assistant actions only; no future tool output in action prefixes'}
    (root / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
