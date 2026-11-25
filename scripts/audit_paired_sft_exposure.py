"""Measure actual RFT/TI token exposure, rather than equating rows with tokens."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paired_method_sft_20261125'


def collect_exposure():
    launch=json.loads((OUT/'launch.json').read_text())
    snapshot=Path(launch['snapshot']);reports={}
    for name in ('ti00','ti25','ti50'):
        data={r['id']:r for r in map(json.loads,(snapshot/name/'train.jsonl').read_text().splitlines())}
        budget=snapshot/name/'dataset_budget_audit.json'
        metrics=OUT/name/'train/metrics.jsonl'
        if not budget.exists()or not metrics.exists():
            reports[name]={'stage':'auditing_or_loading'};continue
        audit=json.loads(budget.read_text())
        lengths={r['id']:r['assistant_tokens']for r in audit['rows']}
        unique_steps={}
        for line in metrics.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            unique_steps[r['step']]=r
        methods=collections.Counter();datasets=collections.Counter();teachers=collections.Counter();visits=collections.Counter()
        for step,r in sorted(unique_steps.items()):
            for sid in r['sample_ids']:
                meta=data[sid]['meta'];n=lengths[sid]
                methods[meta['generation_mode']]+=n
                datasets[meta['dataset']]+=n
                teachers[meta['teacher_backend']]+=n
                visits[sid]+=1
        total=sum(methods.values());last=unique_steps[max(unique_steps)]if unique_steps else{}
        if total!=last.get('supervised_tokens',0):
            raise ValueError('Reconstructed token exposure does not match trainer: '+name)
        reports[name]={'step':last.get('step',0),'total_assistant_tokens':total,
            'method_tokens':dict(methods),'ti_token_fraction':methods['ti']/total if total else None,
            'source_tokens':dict(datasets),'teacher_tokens':dict(teachers),
            'distinct_rows_seen':len(visits),'mean_visits_per_dataset_row':sum(visits.values())/len(data),
            'actual_counter_matches':True,'assistant_token_budget':1000000}
    return reports


def main():
    reports=collect_exposure()
    (OUT/'exposure.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
