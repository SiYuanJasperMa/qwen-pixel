"""Collect real multi-tool editing trajectories; judge before SFT inclusion."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import (TOOLS, RGB_TOOLS, RGB_SYSTEM, assistant_call, copy_source_image, execute_action,
                                  export_sft_steps, initial_messages)
from pixel.teacher_api import MODEL, api_messages, judge_edit, make_client


def build_tasks(count):
    rows = [json.loads(l) for l in (ROOT / 'data/sft_code_coco_v1/train.jsonl').open()]
    scenes = {}
    for row in rows:
        image = Path(row['image'])
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest not in scenes:
            scenes[digest] = []
        instruction = row['instruction'].removeprefix('EDIT: ')
        if 'remove' not in instruction.lower() and instruction not in [r['instruction'] for r in scenes[digest]]:
            scenes[digest].append({'image': str(image), 'instruction': instruction})
        if len([s for s in scenes.values() if s]) >= count:
            break
    selected = [(digest, samples) for digest, samples in scenes.items() if samples][:count]
    tasks = []
    for i, (digest, samples) in enumerate(selected):
        sample = samples[0] if i < 12 else samples[i % len(samples)]
        tasks.append({'id': f'coco_agent_{i:03d}', **sample,
            'scene_sha256': digest, 'split': 'val' if i % 4 == 3 else 'train',
            'source': 'real_COCO_photo_with_edit_instruction; not official MagicBrush evaluation'})
    return tasks


def collect(task, output, max_steps, effort='low', retry_failed=False, system_profile='default'):
    case_root = output / task['id']
    case_root.mkdir(parents=True, exist_ok=True)
    existing = case_root / 'result.json'
    previous = None
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get('system_profile', 'default') != system_profile:
            raise ValueError('Do not mix tool/system profiles in one teacher output directory.')
        if not retry_failed or previous.get('judge', {}).get('pass'):
            return previous
    workspace = case_root / ('retry_' + effort) if previous else case_root
    workspace.mkdir(parents=True, exist_ok=True)
    if workspace != case_root and (workspace / 'result.json').exists():
        return json.loads((workspace / 'result.json').read_text())
    partial = workspace / 'trajectory.json'
    if partial.exists() and not existing.exists():
        archives = workspace / 'interrupted'
        archives.mkdir(exist_ok=True)
        shutil.copyfile(partial, archives / f'trajectory_{time.time_ns()}.json')
    copy_source_image(task['image'], workspace / 'source.png')
    messages = initial_messages(workspace / 'source.png', task['instruction'])
    if system_profile == 'rgb':
        messages[0]['content'] = RGB_SYSTEM
    tools = RGB_TOOLS if system_profile == 'rgb' else TOOLS
    client = make_client()
    turns = []
    complete = False
    for step in range(max_steps):
        start = time.time()
        response = client.chat.completions.create(model=MODEL,
            messages=api_messages(messages), tools=tools, parallel_tool_calls=False,
            reasoning_effort=effort, max_completion_tokens=8192 if effort != 'low' else 5000)
        answer = response.choices[0].message
        if not answer.tool_calls:
            messages.append({'role': 'assistant', 'content': answer.content or ''})
            messages.append({'role': 'user', 'content': 'Execute the edit using the available tools and submit the actual files with finish.'})
            continue
        call = answer.tool_calls[0]
        arguments = json.loads(call.function.arguments)
        action = assistant_call(call.function.name, arguments)
        action['content'] = answer.content or ''
        messages.append(action)
        feedback, passed = execute_action(call.function.name, arguments, workspace, step, system_profile == 'rgb')
        messages.append(feedback)
        turns.append({'step': step, 'action': call.function.name, 'passed': passed,
            'seconds': time.time() - start, 'usage': response.usage.model_dump() if response.usage else None})
        print('TEACHER_ACTION', task['id'], step, call.function.name, passed, flush=True)
        complete = call.function.name == 'finish' and passed
        (workspace / 'trajectory.json').write_text(json.dumps(
            {'task': task, 'teacher_model': MODEL, 'messages': messages, 'turns': turns,
             'complete': complete}, indent=2))
        if complete:
            break
    result = {'id': task['id'], 'split': task['split'], 'complete': complete,
              'steps': len(turns), 'teacher_model': MODEL, 'scene_sha256': task['scene_sha256'],
              'reasoning_effort': effort, 'attempt_directory': str(workspace)}
    result['system_profile'] = system_profile
    if previous:
        result['previous_attempt'] = previous
    if complete:
        submitted = json.loads(messages[-1]['content'])
        result['judge'] = judge_edit(client, workspace / 'source.png',
            workspace / submitted['image_path'], task['instruction'])
        if result['judge']['pass']:
            result['sft_actions'] = export_sft_steps(messages, workspace / 'sft_steps.jsonl',
                task['id'], 'actual_GPT-5.5_tool_rollout_and_real_feedback; judged_pass', tools)
            if workspace != case_root:
                shutil.copyfile(workspace / 'sft_steps.jsonl', case_root / 'sft_steps.jsonl')
    if workspace != case_root:
        (workspace / 'result.json').write_text(json.dumps(result, indent=2))
    existing.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--effort', choices=['low', 'medium', 'high'], default='low')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--tasks', help='Optional task JSONL with id, image, instruction, and train/val split.')
    parser.add_argument('--system-profile', choices=['default', 'rgb'], default='default')
    parser.add_argument('--out', default=str(ROOT / 'data/agent_teacher_pilot_v1'))
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    tasks_file = output / 'tasks.jsonl'
    tasks = [json.loads(l) for l in tasks_file.open()] if tasks_file.exists() else []
    if args.tasks and not tasks:
        tasks = [json.loads(l) for l in Path(args.tasks).open()]
        for task in tasks:
            if task.get('split') not in ('train', 'val'):
                raise ValueError('Teacher SFT construction accepts train/val tasks only.')
            task['scene_sha256'] = hashlib.sha256(Path(task['image']).read_bytes()).hexdigest()
    elif len(tasks) < args.count:
        expanded = build_tasks(args.count)
        tasks = tasks + expanded[len(tasks):]
    if not tasks:
        raise ValueError('No available task images. Provide --tasks with real source files.')
    if any(t.get('evaluation_only')for t in tasks):
        raise ValueError('Evaluation-only records cannot be used for teacher SFT construction.')
    tasks_file.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(collect, task, output, args.max_steps, args.effort, args.retry_failed, args.system_profile): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'id': task['id'], 'split': task['split'], 'complete': False,
                          'infrastructure_error': type(error).__name__}
            results.append(result)
            print(json.dumps(result), flush=True)
            (output / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    for split in ('train', 'val'):
        kept = [r for r in results if r['split'] == split and r.get('judge', {}).get('pass')]
        data = ''.join((output / r['id'] / 'sft_steps.jsonl').read_text() for r in kept)
        (output / (split + '.jsonl')).write_text(data)
    summary = {'tasks': len(tasks), 'execution_complete': sum(r['complete'] for r in results),
        'judge_pass': sum(r.get('judge', {}).get('pass', False) for r in results),
        'train_actions': len((output / 'train.jsonl').read_text().splitlines()),
        'val_actions': len((output / 'val.jsonl').read_text().splitlines()),
        'infrastructure_errors': sum('infrastructure_error' in r for r in results)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('TEACHER_COLLECTION_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
