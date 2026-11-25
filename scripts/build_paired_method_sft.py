"""Source/teacher-matched one-shot RFT/TI supervision comparisons.

Each treatment has exactly one answer per selected task. A TI treatment replaces
the RFT answer; it does not add tasks, duplicate records or expose target inputs.
The comparison is conditional on both construction methods being accepted.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.data_language import english_task, english_supervision
from pixel.source_policy import source_disabled

UNAVAILABLE_EVIDENCE = re.compile(
    r'\b(?:target image|reference image|second image|image\s*2)\b', re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def answer_allowed(row):
    if source_disabled(row):
        return False, 'source_disabled_by_user'
    if not english_task(row) or not english_supervision(row):
        return False, 'non_english'
    answer = json.loads(row['answer'])
    evidence = json.dumps({k: v for k, v in answer.items() if k != 'executable_code'})
    if UNAVAILABLE_EVIDENCE.search(evidence):
        return False, 'teacher_only_evidence_in_student_answer'
    from build_magicbrush_target_informed import looks_hardcoded
    blocked, reason = looks_hardcoded(answer.get('executable_code', ''))
    return not blocked, reason


def variant(pairs, ti_percent):
    """Nested, deterministic treatment assignment with unchanged task order."""
    n = len(pairs)
    order = sorted(range(n), key=lambda i: digest(pairs[i]['key'].encode()))
    selected = set(order[:round(n * ti_percent / 100)])
    return [p['ti' if i in selected else 'rft'] for i, p in enumerate(pairs)]


def freeze(path, contents):
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError('Refusing to overwrite frozen artifact: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pairs', type=int, default=256)
    parser.add_argument('--max-assistant-tokens', type=int, default=6144)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/paired_method_sft_20261125')
    args = parser.parse_args()
    if args.max_pairs < 4:
        raise ValueError('Need at least four pairs.')
    folders = [ROOT/'data/editing_growth_v1', ROOT/'data/editing_growth_v1/gpt62',
               ROOT/'data/editing_growth_v1/ti_balanced_en_v1']
    candidates = collections.defaultdict(dict)
    rejects = collections.Counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    for folder in folders:
        path = folder / 'sft/train.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row['meta']
            if meta.get('split') != 'train' or meta.get('judge_rubric') != 'full_visible_target_v1':
                rejects['split_or_rubric'] += 1
                continue
            allowed, reason = answer_allowed(row)
            if not allowed:
                rejects[reason] += 1
                continue
            if len(tokenizer.encode(row['answer']+'<|im_end|>', add_special_tokens=False)) > args.max_assistant_tokens:
                rejects['assistant_budget_overflow'] += 1
                continue
            key = (meta['dataset'], meta['task_id'], meta['teacher_backend'])
            mode = meta['generation_mode']
            if mode not in ('rft', 'ti'):
                raise ValueError('Unexpected method.')
            # Exported rows are accepted, but keep provenance and scores for audits.
            record_path = folder/'records'/(row['id']+'.json')
            row = {**row, 'meta': {**meta, 'construction_record': str(record_path)}}
            candidates[key][mode] = row
    eval_rows = []
    eval_paths = [ROOT/'data/rl_smoke_en_20261125/tasks_eval64.jsonl',
                  ROOT/'data/editing_growth_v1/tasks_val.jsonl']
    for p in eval_paths:
        if p.exists():
            eval_rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    eval_groups = {r.get('source_group_id') for r in eval_rows}
    rgb_cache = {}
    def rgb_identity(path):
        p = resolve(path)
        key = str(p)
        if key not in rgb_cache:
            with Image.open(p) as im:
                im = im.convert('RGB')
                rgb_cache[key] = digest(str(im.size).encode()+im.tobytes())
        return rgb_cache[key]
    eval_pixels = {rgb_identity(r['image']) for r in eval_rows}
    buckets = collections.defaultdict(list)
    for key, pair in sorted(candidates.items()):
        if set(pair) != {'rft', 'ti'}:
            continue
        r, t = pair['rft'], pair['ti']
        if any(r[k] != t[k] for k in ('system', 'instruction')) or \
                r['meta']['source_group_id'] != t['meta']['source_group_id']:
            rejects['input_mismatch'] += 1
            continue
        rh, th = rgb_identity(r['image']), rgb_identity(t['image'])
        if rh != th:
            rejects['source_pixel_mismatch'] += 1
            continue
        if r['meta']['source_group_id'] in eval_groups or rh in eval_pixels:
            raise ValueError('Evaluation source leaked into paired training data.')
        # Identical path makes processor/image inputs identical in all treatments.
        t = {**t, 'image': r['image']}
        buckets[key[0]].append({'key': '|'.join(key), 'rft': r, 'ti': t, 'rgb_sha256': rh})
    for bucket in buckets.values():
        bucket.sort(key=lambda p: digest(p['key'].encode()))
    selected = []
    # Source-balanced pilot; all eligible pairs remain enumerated in provenance.
    while len(selected) < args.max_pairs and any(buckets.values()):
        for ds in sorted(buckets):
            if buckets[ds] and len(selected) < args.max_pairs:
                selected.append(buckets[ds].pop(0))
    if len(selected) < 4:
        raise ValueError('Insufficient matched accepted pairs.')
    payloads = {}
    variants = {}
    for percent in (0, 25, 50):
        name = f'ti{percent:02d}'
        rows = variant(selected, percent)
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
        payloads[name] = payload
        variants[name] = {'rows': len(rows), 'sha256': digest(payload),
            'method_rows': dict(collections.Counter(r['meta']['generation_mode'] for r in rows)),
            'dataset_methods': dict(collections.Counter(r['meta']['dataset']+'/'+r['meta']['generation_mode'] for r in rows))}
    fingerprint = digest(b''.join(payloads.values())+f'budget_v1:{args.max_assistant_tokens}'.encode())[:12]
    out = args.out / f'pairs_{len(selected)}_{fingerprint}'
    for name, payload in payloads.items():
        freeze(out/name/'train.jsonl', payload)
    evidence = [{k:p[k] for k in ('key','rgb_sha256')} | {
        'rft_id':p['rft']['id'],'ti_id':p['ti']['id'],
        'source_group_id':p['rft']['meta']['source_group_id']} for p in selected]
    manifest = {'interface':'one_shot_source_image_instruction_to_json_python',
        'paired_tasks':len(selected),'variants':variants,'rejected_rows_or_pairs':dict(rejects),
        'source_groups':len({p['rft']['meta']['source_group_id'] for p in selected}),
        'dataset_pairs':dict(collections.Counter(p['rft']['meta']['dataset'] for p in selected)),
        'teacher_pairs':dict(collections.Counter(p['rft']['meta']['teacher_backend'] for p in selected)),
        'validation_source_group_and_rgb_overlap':0,'evaluation_files':list(map(str,eval_paths)),
        'pairs':evidence,'condition':'both methods accepted; no claim of TI rescuing failed tasks',
        'quality':'existing full_visible_target_v1 accepted supervision; needs blinded visual audit',
        'ti_ratio_is_row_fraction':'assistant-token fractions must be measured from actual training logs',
        'max_assistant_tokens':args.max_assistant_tokens,
        'sources':list(map(str,folders))}
    freeze(out/'manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps({'out':str(out), **{k:v for k,v in manifest.items() if k != 'pairs'}}, indent=2))


if __name__ == '__main__':
    main()
