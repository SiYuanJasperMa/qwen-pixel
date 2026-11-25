"""Fetch the original COCO photos for the same image-disjoint editing tasks."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def download(task, original, out):
    key = Path(task['image']).name.removesuffix('_source.png')
    scene = str(original[key]['meta']['scene_id'])
    name = f'COCO_train2014_{int(scene):012d}.jpg'
    path = out / 'images' / name
    url = 'https://s3.amazonaws.com/images.cocodataset.org/train2014/' + name
    if not path.exists():
        for attempt in range(3):
            try:
                with urlopen(url, timeout=30) as response:
                    payload = response.read()
                with Image.open(__import__('io').BytesIO(payload)) as im:
                    im.verify()
                path.write_bytes(payload)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with Image.open(path) as im:
        dimensions = list(im.size)
    return {**task, 'image': str(path), 'original_scene_id': scene,
            'scene_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_resolution': dimensions, 'source_url': url,
            'source': 'original_native_resolution_COCO_train2014_photo_with_RefCOCO_derived_edit_instruction'}


def main():
    out = ROOT / 'data/agent_native_v4'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    original = {r['id']: r for r in (json.loads(l) for l in
        (ROOT / 'data/edit_tasks_coco_v1/tasks.jsonl').open())}
    tasks = [json.loads(l) for l in (ROOT / 'data/agent_teacher_scale_v2/tasks.jsonl').open()]
    records = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(download, task, original, out): task for task in tasks}
        for future in concurrent.futures.as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
                records.append(result)
                print('NATIVE_PHOTO', result['id'], result['source_resolution'], flush=True)
            except Exception as error:
                failures.append({'id': task['id'], 'error': type(error).__name__})
    records.sort(key=lambda r: r['id'])
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    summary = {'photos': len(records), 'train': sum(r['split'] == 'train' for r in records),
        'val': sum(r['split'] == 'val' for r in records), 'failures': failures,
        'native_resolution_preserved': True}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('NATIVE_TASKS_READY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
