"""Public aggregate evidence: English sample rows and paired one-shot RL results."""
import collections
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.data_language import english_task, english_supervision


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def collect():
    paths = [ROOT/'data/star_clean_v2/sft_mixed_aug/train.jsonl',
             ROOT/'data/editing_growth_v1/sft/train.jsonl',
             ROOT/'data/editing_growth_v1/gpt62/sft/train.jsonl',
             ROOT/'data/editing_growth_v1/ti_balanced_en_v1/sft/train.jsonl',
             ROOT/'data/editing_growth_v1/ti_night_sol_v2/sft/train.jsonl']
    counts = collections.Counter()
    teachers = collections.Counter()
    tasks = set()
    for p in paths:
        for row in rows(p):
            if not english_task(row) or not english_supervision(row):
                continue
            m = row['meta']
            ds = m.get('dataset') or next((d for d in ('magicbrush','ultraedit','sciforma')
                                           if m['task_id'].startswith(d)), 'unknown')
            counts[(ds, m['generation_mode'])] += 1
            teachers[(m.get('teacher_backend') or 'unrecorded', m['generation_mode'])] += 1
            tasks.add(m['task_id'])
    inventory = {'row_policy':'accepted English exports; includes distinct teacher and augmentation rows',
        'unique_tasks':len(tasks),'datasets':[{'dataset':d,'rft':counts[(d,'rft')],
            'ti':counts[(d,'ti')],'total':counts[(d,'rft')]+counts[(d,'ti')]}
            for d in sorted({d for d, _ in counts})],
        'teachers':[{'teacher':t,'rft':teachers[(t,'rft')],'ti':teachers[(t,'ti')]}
            for t in sorted({t for t, _ in teachers})]}
    inventory['rft'] = sum(c for (d,m),c in counts.items() if m == 'rft')
    inventory['ti'] = sum(c for (d,m),c in counts.items() if m == 'ti')
    inventory['total'] = inventory['rft'] + inventory['ti']
    folder = ROOT/'outputs/rl_smoke_en_20261125'
    rl = {}
    for stage in ('pre','post'):
        result = [r for p in sorted((folder/stage).glob('shard*/results.jsonl')) for r in rows(p)]
        ids = [r['id'] for r in result]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate RL evaluation task.')
        rl[stage] = {'tasks':len(result),'executed':sum((r.get('execution') or {}).get('status')=='passed' for r in result),
            'pass':sum((r.get('judge') or {}).get('pass') is True for r in result),
            'judge_pending':sum(bool(r.get('judge_pending')) for r in result)}
        rl[stage+'_cases'] = {r['id']:{'pass':(r.get('judge') or {}).get('pass') is True,
            'instruction':r['instruction'],'serializer':r.get('serializer_version')} for r in result}
    a,b = rl['pre_cases'],rl['post_cases']
    if a and b:
        if a.keys()!=b.keys() or any(a[k]['instruction']!=b[k]['instruction'] or
                                      a[k]['serializer']!=b[k]['serializer'] for k in a):
            raise ValueError('RL evaluation prompts are not matched.')
        rl.update(matched_tasks=len(a),gained=sum(not a[k]['pass'] and b[k]['pass'] for k in a),
                  lost=sum(a[k]['pass'] and not b[k]['pass'] for k in a))
    summary_path = folder/'train/summary.json'
    if summary_path.exists():
        training=json.loads(summary_path.read_text())
        rl.update(optimizer_updates=training['optimizer_updates'],
            frozen_reference_unchanged=training['initial_reference_sha256']==training['final_reference_sha256'])
    rl['measurement']='custom full_visible_target_v1 image judge; not official benchmark metrics'
    rl['limitation']='single seed and legacy greedy decoding; repeat inference before attributing the gain to RL'
    return {'updated':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            'inventory':inventory,'rl':rl}


if __name__ == '__main__':
    evidence=collect()
    out=ROOT/'outputs/current_editing_evidence_20261125.json'
    out.write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'inventory':evidence['inventory'],'rl':{k:v for k,v in evidence['rl'].items()if not k.endswith('_cases')}},indent=2))
