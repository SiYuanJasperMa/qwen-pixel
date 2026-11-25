"""Use one complete, accepted native-photo trajectory for adapter-scope smoke."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/sft_clean_mask_v4_raw';out=ROOT/'data/sft_attention_smoke_v1'
    out.mkdir(exist_ok=True)
    if (out/'train.jsonl').exists():raise ValueError('Attention smoke data already prepared; do not replace a running experiment.')
    rows=[json.loads(l)for l in(data/'train_verified.jsonl').open()]
    identity=rows[0]['meta']['trajectory_id']
    selected=[r for r in rows if r['meta']['trajectory_id']==identity]
    if len(selected)!=4:raise ValueError('Need a complete real training trajectory.')
    task=next(t for t in(json.loads(l)for l in(data/'tasks.jsonl').open())if t['id']==identity)
    task['target_bbox_pixels']=json.loads((data/'executions'/identity/'region.json').read_text())['source_bbox_pixels']
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in selected))
    (out/'tasks.jsonl').write_text(json.dumps(task)+'\n')
    print('ATTENTION_SCOPE_ONE_NATIVE_TRAJECTORY_READY',identity,flush=True)


if __name__=='__main__':main()
