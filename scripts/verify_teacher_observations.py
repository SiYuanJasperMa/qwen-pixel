"""Replay teachers and retain only certified, immutable observation histories."""
import argparse
import concurrent.futures
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import execute_action, export_sft_steps, initial_messages


def equal_pixels(first, second):
    with Image.open(first) as a, Image.open(second) as b:
        return np.array_equal(np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB')))


def verify(task, source_root, destination):
    selected_file = source_root / task['id'] / 'result.json'
    selected = json.loads(selected_file.read_text())
    if not selected.get('judge', {}).get('pass'):
        return {'id': task['id'], 'split': task['split'], 'kept': False, 'reason': 'edit_quality_failed'}
    old_workspace = Path(selected.get('attempt_directory', source_root / task['id']))
    native = json.loads((old_workspace / 'trajectory.json').read_text())
    workspace = destination / task['id']
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task['image'], workspace / 'source.png')
    frozen_input = initial_messages(workspace / 'source.png', task['instruction'])[1]['content'][0]['image']
    messages = copy.deepcopy(native['messages'])
    for message in messages:
        if message['role'] == 'user' and isinstance(message['content'], list):
            for block in message['content']:
                if block.get('type') == 'image':
                    if not equal_pixels(block['image'], frozen_input):
                        return {'id': task['id'], 'split': task['split'], 'kept': False, 'reason': 'initial_source_changed'}
                    block['image'] = frozen_input
    pending = None
    step = 0
    mismatches = []
    for message in messages:
        if message['role'] == 'assistant' and message.get('tool_calls'):
            pending = message['tool_calls'][0]['function']
        elif message['role'] == 'tool' and pending:
            feedback, passed = execute_action(pending['name'], pending['arguments'], workspace, step)
            if pending['name'] == 'view_image':
                if not passed:
                    mismatches.append({'step': step, 'reason': 'view_replay_failed'})
                elif isinstance(message['content'], list):
                    old_images = [b for b in message['content'] if b.get('type') == 'image']
                    new_images = [b for b in feedback['content'] if b.get('type') == 'image']
                    for old, new in zip(old_images, new_images):
                        if not equal_pixels(old['image'], new['image']):
                            mismatches.append({'step': step, 'reason': 'historical_image_was_overwritten_or_nondeterministic'})
                        old['image'] = new['image']
            elif pending['name'] == 'finish':
                if not passed or not equal_pixels(workspace / pending['arguments']['image_path'],
                                                 old_workspace / pending['arguments']['image_path']):
                    mismatches.append({'step': step, 'reason': 'final_pixels_not_reproducible'})
            pending = None
            step += 1
    result = {'id': task['id'], 'split': task['split'], 'kept': not mismatches,
              'mismatches': mismatches, 'judge': selected['judge']}
    if not mismatches:
        count = export_sft_steps(messages, workspace / 'sft_steps.jsonl', task['id'],
            'actual_teacher_actions_and_tool_text; each_static_image_verified_by_deterministic_replay_and_frozen; mutated_histories_excluded')
        result['assistant_actions'] = count
        (workspace / 'trajectory_verified.json').write_text(json.dumps(
            {'task': task, 'messages': messages, 'observation_replay_verified': True}, indent=2))
    (workspace / 'verification.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = Path(args.source); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in (source / 'tasks.jsonl').open()]
    shutil.copyfile(source / 'tasks.jsonl', out / 'tasks.jsonl')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(verify, t, source, out): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try: result = future.result()
            except Exception as error: result = {'id': task['id'], 'split': task['split'], 'kept': False, 'reason': type(error).__name__}
            results.append(result)
            print('OBSERVATION_AUDIT', json.dumps(result), flush=True)
    for split in ('train', 'val'):
        data = ''.join((out / r['id'] / 'sft_steps.jsonl').read_text()
                       for r in results if r['split'] == split and r['kept'])
        (out / (split + '.jsonl')).write_text(data)
    summary = {'tasks': len(tasks), 'verified_trajectories': sum(r['kept'] for r in results),
        'unsafe_observation_histories': sum(bool(r.get('mismatches')) for r in results),
        'train_actions': len((out / 'train.jsonl').read_text().splitlines()),
        'val_actions': len((out / 'val.jsonl').read_text().splitlines()),
        'results': results}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('OBSERVATION_AUDIT_COMPLETE', json.dumps({k:v for k,v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
