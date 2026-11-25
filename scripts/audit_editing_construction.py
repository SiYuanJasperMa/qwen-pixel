"""Measure accepted supervision and paired target fidelity from current exports.

This reads exported English rows only, and never modifies a training snapshot.
Pixel-gain diagnostics are not semantic success or official benchmark scores.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.data_language import english_task, english_supervision

FOLDERS = [ROOT/'data/editing_growth_v1', ROOT/'data/editing_growth_v1/gpt62',
           ROOT/'data/editing_growth_v1/ti_balanced_en_v1',
           ROOT/'data/editing_growth_v1/ti_night_sol_v2']


def collect():
    groups = collections.defaultdict(list)
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    seen_ids, tasks, sources, sources_read = collections.Counter(), set(), set(), []
    rejected = collections.Counter()
    duplicate_answers = collections.Counter()
    for folder in FOLDERS:
        path = folder/'sft/train.jsonl'
        if not path.is_file():
            continue
        payload = path.read_bytes()
        sources_read.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not english_task(row) or not english_supervision(row):
                rejected['non_english'] += 1
                continue
            seen_ids[row['id']] += 1
            meta = row['meta']
            if meta.get('split') != 'train':
                raise ValueError('Non-training row in production export')
            ds, teacher, mode = meta['dataset'], meta['teacher_backend'], meta['generation_mode']
            task = (ds, meta['task_id'])
            tasks.add(task); sources.add(meta['source_group_id'])
            groups[(ds, teacher, mode)].append(row)
            # JSON field ordering alone does not create distinct supervision.
            answer = json.dumps(json.loads(row['answer']), sort_keys=True, ensure_ascii=False)
            duplicate_answers[(task, meta['source_group_id'], row['system'], row['instruction'],
                               hashlib.sha256(answer.encode()).hexdigest())] += 1
            key = (ds, meta['task_id'], teacher, meta['source_group_id'],
                   row['system'], row['instruction'])
            pairs[key][mode].append(row)
    by_dataset, by_teacher, paired = [], [], []
    for ds in sorted({k[0] for k in groups}):
        rr = [r for (d, _, m), rows in groups.items() if d == ds and m == 'rft' for r in rows]
        tt = [r for (d, _, m), rows in groups.items() if d == ds and m == 'ti' for r in rows]
        by_dataset.append({'dataset': ds, 'rft': len(rr), 'ti': len(tt),
                           'total': len(rr)+len(tt),
                           'unique_tasks': len({r['meta']['task_id'] for r in rr+tt})})
    for teacher in sorted({k[1] for k in groups}):
        by_teacher.append({'teacher': teacher, **{mode: sum(len(rows) for (_, t, m), rows in groups.items()
                                                        if t == teacher and m == mode)
                                                 for mode in ('rft', 'ti')}})
    fidelity = []
    for (ds, teacher, mode), rr in sorted(groups.items()):
        gain = [r['meta']['pixel_metrics']['gain'] for r in rr
                if isinstance((r['meta'].get('pixel_metrics') or {}).get('gain'), (float, int))]
        fidelity.append({'dataset': ds, 'teacher': teacher, 'mode': mode, 'rows': len(rr),
                         'gain_measured': len(gain), 'positive_l1_gain': sum(x > 0 for x in gain),
                         'mean_l1_gain': statistics.mean(gain) if gain else None,
                         'median_l1_gain': statistics.median(gain) if gain else None})
    for key, modes in sorted(pairs.items()):
        if not modes.get('rft') or not modes.get('ti'):
            continue
        # Deterministic first candidate rather than choosing by the target metric.
        r, t = [sorted(modes[m], key=lambda row: row['id'])[0] for m in ('rft', 'ti')]
        rm, tm = [row['meta'].get('pixel_metrics') or {} for row in (r, t)]
        measured = all(isinstance(m.get('l1'), (int, float)) for m in (rm, tm))
        same_noop = measured and all(isinstance(m.get('no_op_l1'), (int, float)) for m in (rm, tm)) \
            and abs(rm['no_op_l1']-tm['no_op_l1']) < 1e-6
        paired.append({'dataset': key[0], 'task_id': key[1], 'teacher': key[2],
                       'rft_id': r['id'], 'ti_id': t['id'],
                       'metric_reference_consistent': same_noop,
                       'ti_l1_improvement': rm['l1']-tm['l1'] if same_noop else None})
    delta = [p['ti_l1_improvement'] for p in paired if p['ti_l1_improvement'] is not None]
    return {'updated': dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            'scope': 'new English growth exports; excludes historical baseline and live SFT snapshots',
            'accepted_rows': sum(seen_ids.values()), 'unique_tasks': len(tasks), 'source_groups': len(sources),
            'candidate_id_collision_excess_rows': sum(n-1 for n in seen_ids.values()),
            'candidate_id_policy': 'Original rescue/enrichment workers can reuse an ID. Namespace by export origin before freezing or ID-keyed sampling; preserve original ID as provenance.',
            'duplicate_supervision_excess_rows': sum(n-1 for n in duplicate_answers.values()),
            'by_dataset': by_dataset, 'by_teacher': by_teacher, 'target_fidelity': fidelity,
            'matched_pairs': len(paired), 'metric_consistent_pairs': len(delta),
            'ti_l1_better_pairs': sum(x > 0 for x in delta),
            'mean_paired_ti_l1_improvement': statistics.mean(delta) if delta else None,
            'paired_selection': 'same task/teacher/source group/system/instruction; both accepted; first candidate by ID',
            'limitations': ['Actual pixel equality of source/target is not re-audited here; no-op metric agreement is only a diagnostic.',
                           'Matched accepted pairs are selected for success; this is not a rescue-rate comparison.',
                           'Aligned resized RGB L1 is not semantic success or official benchmark evaluation.',
                           'Semantic pass derives from the existing full_visible_target_v1 VLM filter; requires blinded human calibration.'],
            'pairs': paired, 'rejected': dict(rejected), 'exports': sources_read}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/editing_construction_audit_20261125.json')
    args = parser.parse_args()
    report = collect()
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'exports', 'target_fidelity')}, indent=2))
