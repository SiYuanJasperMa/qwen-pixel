"""Repartition existing SFT and build TI rescue tasks from observed failures.

Reads original source-only attempts, not just the accepted-trajectory pool.
Every teacher/mode/turn on a source group inherits the same split.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.data_lineage import canonical_task, source_group, split_of_group


def read(path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def reconcile_content(tasks, sft, groups):
    """Union original-image groups across datasets by source/target RGB content."""
    parents, hashes, owners = {}, {}, {}
    def root(g):
        parents.setdefault(g, g)
        while parents[g] != g:
            parents[g] = parents[parents[g]]; g = parents[g]
        return g
    def join(a, b):
        a, b = root(a), root(b)
        if a != b: parents[max(a, b)] = min(a, b)
    def digest(path):
        p = Path(path)
        if not p.is_absolute(): p = ROOT / p
        key = str(p)
        if key not in hashes:
            with Image.open(p) as im:
                pixels = im.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
                hashes[key] = hashlib.sha256(pixels.tobytes()).hexdigest()
        return hashes[key]
    print('CONTENT_GROUPING_START', len(tasks), flush=True)
    for i, t in enumerate(tasks.values()):
        g = groups[t['id']]
        for path in (t['image'], t['target_image']):
            h = digest(path)
            if h in owners: join(g, owners[h])
            else: owners[h] = g
        if (i + 1) % 2000 == 0: print('CONTENT_GROUPING', i + 1, flush=True)
    for r in sft:
        g = groups[r['meta']['task_id']]
        h = digest(r['image'])
        if h in owners: join(g, owners[h])
        else: owners[h] = g
    blocked = set()
    for t in read(ROOT / 'data/magicbrush_official_dev_v1/tasks.jsonl'):
        for path in (t.get('image'), t.get('target_image')):
            if path and Path(path).is_file():
                h = digest(path)
                if h in owners: blocked.add(root(owners[h]))
    for sid in groups: groups[sid] = root(groups[sid])
    blocked = {root(g) for g in blocked}
    excluded = {sid for sid, g in groups.items() if g in blocked}
    tasks = {sid: t for sid, t in tasks.items() if sid not in excluded}
    sft = [r for r in sft if r['meta']['task_id'] not in excluded]
    for r in sft:
        g = groups[r['meta']['task_id']]
        r['meta'].update(source_group_id=g, split=split_of_group(g))
    return tasks, sft, {'normalized_rgb_images_hashed': len(hashes),
                       'official_dev_content_overlap_excluded_tasks': len(excluded)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'data/star_clean_v1'))
    p.add_argument('--exact-content-grouping', action='store_true')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    groups, sft, seen = {}, [], set()
    for split in ('train', 'val'):
        for r in read(ROOT / 'data/sft_all_v14' / f'{split}.jsonl'):
            sid = canonical_task(r['id'])
            if sid not in groups:
                groups[sid] = source_group(sid, r['image'])
            mode = 'ti' if r.get('meta', {}).get('target_informed') else 'rft'
            digest = hashlib.sha256(r['answer'].encode()).hexdigest()[:12]
            key = (sid, mode, digest)
            if key in seen:
                continue
            seen.add(key)
            meta = {**r.get('meta', {}), 'task_id': sid, 'source_group_id': groups[sid],
                    'generation_mode': mode, 'split': split_of_group(groups[sid])}
            sft.append({**r, 'id': sid + '__' + mode + '_' + digest, 'meta': meta})
    sources = [ROOT / 'data/teacher_magicbrush_scale_v1/attempts.jsonl',
               ROOT / 'data/teacher_magicbrush_merged_v1/judged.jsonl']
    sources += sorted((ROOT / 'data').glob('ultraedit_teacher_*/judged.jsonl'))
    sources += sorted((ROOT / 'data').glob('sciforma_teacher_*/judged.jsonl'))
    tasks, successes, failures, missing = {}, set(), set(), 0
    for path in sources:
        for r in read(path):
            sid = canonical_task(r['id'])
            src = Path(r.get('source_image', ''))
            tgt = Path(r.get('target_image', ''))
            if not src.is_absolute(): src = ROOT / src
            if not tgt.is_absolute(): tgt = ROOT / tgt
            if not src.is_file() or not tgt.is_file():
                missing += 1; continue
            dataset = 'magicbrush' if sid.startswith('magicbrush_') else sid.split('_')[0]
            if sid not in groups: groups[sid] = source_group(sid, src)
            group = groups[sid]
            if sid not in tasks:
                tasks[sid] = {'id': sid, 'dataset': dataset, 'instruction': r['instruction'],
                    'image': str(src), 'source_image': str(src), 'target_image': str(tgt),
                    'mask_image': r.get('mask_image'), 'source_group_id': group,
                    'split': split_of_group(group), 'original_split': 'train'}
            elif tasks[sid]['instruction'] != r['instruction']:
                raise ValueError('Conflicting instructions for task: ' + sid)
            judge = r.get('judge') or {}
            if judge.get('pass') is True and r.get('code'):
                successes.add(sid)
            elif judge.get('pass') is False or r.get('status') in ('failed', 'timeout', 'judge_fail'):
                failures.add(sid)
    content_stats = {}
    if args.exact_content_grouping:
        tasks, sft, content_stats = reconcile_content(tasks, sft, groups)
    for t in tasks.values():
        t.update(source_group_id=groups[t['id']], split=split_of_group(groups[t['id']]))
        t['rft_success_observed'] = t['id'] in successes
        t['rft_failure_observed'] = t['id'] in failures
        t['ti_rescue_eligible'] = t['id'] in failures and t['id'] not in successes
    # Original dataset image IDs must not overlap the official MagicBrush dev.
    official = {canonical_task(r['id']).split('_')[2]
                for r in read(ROOT / 'data/magicbrush_official_dev_v1/tasks.jsonl')}
    overlaps = [t['id'] for t in tasks.values() if t['dataset'] == 'magicbrush'
                and t['id'].split('_')[2] in official]
    if overlaps: raise ValueError('Official dev image-ID overlap: ' + str(overlaps[:5]))
    stats = {'source_files': [str(p) for p in sources], 'missing_image_records': missing,
             'content_audit': content_stats,
             'official_dev_image_id_overlap': len(overlaps), 'buckets': {}, 'tasks': {}}
    for split in ('train', 'val'):
        subset = [r for r in sft if r['meta']['split'] == split]
        rft = [r for r in subset if r['meta']['generation_mode'] == 'rft']
        write(out / 'sft_mixed' / f'{split}.jsonl', subset)
        write(out / 'sft_rft' / f'{split}.jsonl', rft)
        ts = [t for t in tasks.values() if t['split'] == split]
        write(out / f'tasks_{split}.jsonl', ts)
        rescue = [t for t in ts if t['ti_rescue_eligible']]
        write(out / f'failed_{split}.jsonl', rescue)
        stats['buckets'][split] = dict(Counter(r['meta']['generation_mode'] for r in subset))
        stats['tasks'][split] = {'total': len(ts), 'failed_rescue_eligible': len(rescue),
                               'datasets': dict(Counter(t['dataset'] for t in ts))}
    train_groups = {r['meta']['source_group_id'] for r in sft if r['meta']['split'] == 'train'}
    val_groups = {r['meta']['source_group_id'] for r in sft if r['meta']['split'] == 'val'}
    assert not train_groups & val_groups
    stats['train_val_source_group_overlap'] = 0
    write(out / 'rl_smoke_tasks.jsonl', sorted([t for t in tasks.values()
          if t['split'] == 'train' and t['rft_success_observed'] and t['dataset'] == 'magicbrush'],
          key=lambda t: (len(t['instruction']), t['id']))[:32])
    (out / 'manifest.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
