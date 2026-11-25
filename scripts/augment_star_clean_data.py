"""Add fresh verified RFT retries and TI-only rescues to the clean snapshots."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_magicbrush_target_informed import looks_hardcoded
from build_teacher_magicbrush_inpaint import PROMPT, SYSTEM


def read(p):
    with p.open() as f: return [json.loads(l) for l in f if l.strip()]


def main():
    clean = ROOT / 'data/star_clean_v2'
    tasks = {r['id']: r for split in ('train', 'val') for r in read(clean / f'tasks_{split}.jsonl')}
    rft = []
    for folder in ('star_rft_retry_gpt55_v1', 'star_rft_retry_gpt61_v1'):
        rft.extend(read(ROOT / 'data' / folder / 'judged.jsonl'))
    ti = []
    for folder in ('star_ti_rescue_gpt55_v1', 'star_ti_rescue_gpt61_v1'):
        ti.extend(read(ROOT / 'data' / folder / 'judged.jsonl'))
    def good(r):
        return (r.get('judge') or {}).get('pass') and r.get('code') and not looks_hardcoded(r['code'])[0]
    rft = [r for r in rft if good(r)]
    succeeded = {r['task_id'] for r in rft}
    ti = [r for r in ti if good(r) and r['task_id'] not in succeeded]
    def row(r):
        t = tasks[r['task_id']]
        answer = {**r['reasoning'], 'executable_code': r['code'],
                  'requires_inpaint': r.get('requires_inpaint', False)}
        return {'id': r['id'], 'task': r['dataset'] + '_oneshot', 'system': SYSTEM,
                'instruction': PROMPT.format(instruction=r['instruction']),
                'image': r['source_image'], 'answer': json.dumps(answer, ensure_ascii=False),
                'meta': {'task_id': t['id'], 'source_group_id': t['source_group_id'],
                         'split': t['split'], 'instruction': t['instruction'],
                         'teacher_backend': r['teacher'], 'generation_mode': r['generation_mode'],
                         'target_informed': r['generation_mode'] == 'ti',
                         'construction_origin': 'ti_failure_rescue' if r['generation_mode'] == 'ti' else 'rft_retry'}}
    rft, ti = [row(r) for r in rft], [row(r) for r in ti]
    report = {'new_rft_rows': len(rft), 'new_ti_only_rescues': len(ti), 'datasets': {}}
    for name in ('sft_rft', 'sft_mixed'):
        out = clean / (name + '_aug'); out.mkdir(exist_ok=True)
        counts = {}
        for split in ('train', 'val'):
            rows = read(clean / name / f'{split}.jsonl')
            rows += [r for r in rft if r['meta']['split'] == split]
            if name == 'sft_mixed': rows += [r for r in ti if r['meta']['split'] == split]
            seen, kept = set(), []
            for r in rows:
                key = (r['meta']['task_id'], r['meta']['generation_mode'],
                       hashlib.sha256(r['answer'].encode()).hexdigest())
                if key not in seen: seen.add(key); kept.append(r)
            (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept))
            counts[split] = len(kept)
        report['datasets'][name] = counts
    (clean / 'augmentation_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
