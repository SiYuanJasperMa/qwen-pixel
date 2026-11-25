"""Freeze a deduplicated growth snapshot while preserving existing comparisons."""
import hashlib
import collections
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from continuous_editing_teachers import rows,atomic_json
from pixel.persistent_io import atomic_text
sys.path.insert(0,str(ROOT))
from pixel.data_language import english_task,english_supervision
from pixel.data_quality import review_registry,snapshot_quality_reason


def namespaced_candidate(row, export_path):
    """Keep independently generated answers with a reused legacy ID addressable."""
    original_id = row['id']
    answer = json.dumps(json.loads(row['answer']), sort_keys=True, ensure_ascii=False)
    identity = json.dumps([str(export_path), original_id, answer], ensure_ascii=False).encode()
    return {**row, 'id': original_id+'__export_'+hashlib.sha256(identity).hexdigest()[:16],
            'meta': {**row['meta'], 'original_candidate_id': original_id,
                     'construction_export': str(export_path)}}


def freeze(path, payload):
    if path.exists():
        if path.read_text() != payload:
            raise ValueError('Refusing to overwrite frozen snapshot: '+str(path))
    else:
        atomic_text(path, payload)


def main():
    base=ROOT/'data/star_clean_v2/sft_mixed_aug/train.jsonl'
    growth=ROOT/'data/editing_growth_v1/sft/train.jsonl'
    gpt62=ROOT/'data/editing_growth_v1/gpt62/sft/train.jsonl'
    ti=ROOT/'data/editing_growth_v1/ti_balanced_en_v1/sft/train.jsonl'
    night_ti=ROOT/'data/editing_growth_v1/ti_night_sol_v2/sft/train.jsonl'
    base_rows,growth_rows,gpt62_rows=rows(base),rows(growth),rows(gpt62)
    ti_rows=rows(ti)
    night_ti_rows=rows(night_ti)
    records=[namespaced_candidate(r,path)for path,rr in
             [(base,base_rows),(growth,growth_rows),(gpt62,gpt62_rows),(ti,ti_rows),(night_ti,night_ti_rows)]
             for r in rr]
    validation=[r for r in rows(ROOT/'data/star_clean_v2/sft_mixed_aug/val.jsonl') if english_task(r) and english_supervision(r)]
    val_groups={r['meta']['source_group_id']for r in validation}
    kept=[];seen=set()
    review=review_registry();quality_records={};quality_excluded=collections.Counter()
    for path,rr in [(base,base_rows),(growth,growth_rows),(gpt62,gpt62_rows),(ti,ti_rows),(night_ti,night_ti_rows)]:
        ids={r['id']for r in rr if(r['meta'].get('dataset')=='instructpix2pix' or r['meta'].get('task_id','').startswith('instructpix2pix'))}
        judged=path.parent.parent/'judged.jsonl'
        if not ids or not judged.exists():continue
        with judged.open()as handle:
            for line in handle:
                if 'instructpix2pix'not in line:continue
                record=json.loads(line)
                if record.get('id')in ids:
                    quality_records[(str(path),record['id'])]={'judge':record.get('judge'),
                        'pipeline_qualified':record.get('pipeline_qualified'),
                        'code_sha256':hashlib.sha256((record.get('code')or'').encode()).hexdigest()}
    for r in records:
        if not english_task(r) or not english_supervision(r):continue
        meta=r['meta']
        reason=snapshot_quality_reason(r,review,quality_records.get((meta['construction_export'],meta['original_candidate_id'])))
        if reason:
            quality_excluded[reason]+=1;continue
        if meta['source_group_id']in val_groups:raise ValueError('Validation source group in growth training data')
        canonical=json.dumps(json.loads(r['answer']),sort_keys=True,ensure_ascii=False)
        key=(meta['task_id'],meta['generation_mode'],hashlib.sha256(canonical.encode()).hexdigest())
        if key in seen:continue
        seen.add(key);kept.append(r)
    train_payload=''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in kept)
    val_payload=''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in validation)
    digest=hashlib.sha256((train_payload+val_payload).encode()).hexdigest()[:12]
    out=ROOT/f'data/editing_growth_v1/snapshots/mixed_{len(kept)}_{digest}'
    out.mkdir(parents=True,exist_ok=True)
    freeze(out/'train.jsonl',train_payload)
    freeze(out/'val.jsonl',val_payload)
    report={'rows':len(kept),'base_rows':len(base_rows),
            'growth_rows':len(growth_rows)+len(gpt62_rows)+len(ti_rows)+len(night_ti_rows),'gpt62_input_rows':len(gpt62_rows),
            'independent_ti_input_rows':len(ti_rows),'language_policy':'english_only',
            'night_ti_input_rows':len(night_ti_rows),
            'candidate_id_policy':'export_origin_and_answer_hash_v1',
            'quality_policy':'manual review exclusions; InstructPix2Pix qualified code-matched visual-quality>=4 and preservation>=4',
            'quality_excluded_rows':dict(quality_excluded),
            'train_sha256':hashlib.sha256(train_payload.encode()).hexdigest(),
            'val_sha256':hashlib.sha256(val_payload.encode()).hexdigest(),
            'dataset_modes':dict(collections.Counter(r['meta'].get('dataset',r['task'])+'/'+r['meta']['generation_mode']for r in kept)),
            'source_groups':len({r['meta']['source_group_id']for r in kept}),
            'validation_source_group_overlap':0,'sources':[str(base),str(growth)]+([str(gpt62)] if gpt62_rows else [])+([str(ti)] if ti_rows else [])+([str(night_ti)] if night_ti_rows else []),
            'created':time.time()}
    if(out/'manifest.json').exists():
        saved=json.loads((out/'manifest.json').read_text())
        if any(saved.get(k)!=report[k]for k in ('train_sha256','val_sha256','candidate_id_policy')):
            raise ValueError('Frozen snapshot manifest identity mismatch')
        report=saved
    else:atomic_json(out/'manifest.json',report)
    print(json.dumps({'snapshot':str(out),**report}),flush=True)


if __name__=='__main__':main()
