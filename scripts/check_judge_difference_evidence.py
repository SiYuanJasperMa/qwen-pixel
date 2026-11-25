"""Check the evaluator on real subtle edits and known failed model outputs."""
import concurrent.futures
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.teacher_api import make_client,judge_edit


def main():
    out=ROOT/'outputs/sft_audit/judge_delta_controls_v2';out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/sft_clean_mask_v3'
    tasks={t['id']:t for t in(json.loads(l)for l in(data/'tasks.jsonl').open())}
    cases=[]
    for identity in ('clean_551542_418300_brighten','clean_531956_1162687_brighten','clean_542011_2159507_brighten'):
        folder=data/'executions'/identity
        box=json.loads((folder/'region.json').read_text())['source_bbox_pixels']
        cases.append((identity,folder/'source.png',folder/'edited.png',tasks[identity]['instruction'],box,'real_subtle_edit'))
    development={t['id']:t for t in(json.loads(l)for l in(ROOT/'data/sft_clean_mask_v2/development_tasks.jsonl').open())}
    for identity in ('clean_development_00','clean_development_01','clean_development_02'):
        folder=ROOT/'outputs/sft_clean_mask_v2/development_400'/identity;t=development[identity]
        cases.append((identity,folder/'source.png',folder/'edited.png',t['instruction'],t['target_bbox_pixels'],'known_bad_model_edit'))
    first=cases[0]
    cases.append(('literal_noop',first[1],first[1],first[3],first[4],'literal_noop'))
    identity='clean_552184_491711_brighten';folder=data/'executions'/identity
    box=json.loads((folder/'region.json').read_text())['source_bbox_pixels']
    with Image.open(folder/'source.png')as image:source=np.asarray(image.convert('RGB'))
    x0,y0,x1,y1=box;partial=source.copy();limit=y0+max(1,(y1-y0)//4)
    partial[y0:limit,x0:x1]=np.rint(255*np.power(source[y0:limit,x0:x1].astype(float)/255,.65)).astype(np.uint8)
    partial_path=out/'synthetic_partial_target.png';Image.fromarray(partial).save(partial_path)
    cases.append(('synthetic_partial_target',folder/'source.png',partial_path,tasks[identity]['instruction'],box,'known_partial_target'))
    whole=np.rint(255*np.power(source.astype(float)/255,.65)).astype(np.uint8)
    whole_path=out/'synthetic_global_spill.png';Image.fromarray(whole).save(whole_path)
    cases.append(('synthetic_global_spill',folder/'source.png',whole_path,tasks[identity]['instruction'],box,'known_global_spill'))
    def review(case):
        identity,source,edited,instruction,box,kind=case
        judgment=judge_edit(make_client(),source,edited,instruction,True,box)
        result={'id':identity,'kind':kind,'judge':judgment,'actor_regenerated':False}
        (out/(identity+'.json')).write_text(json.dumps(result,indent=2));return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
        for result in pool.map(review,cases):
            results.append(result);print(json.dumps(result),flush=True)
    summary={'cases':len(results),'bad_or_noop_accepted':sum(r['judge']['pass']for r in results if r['kind']!='real_subtle_edit'),
        'subtle_edit_accepted':sum(r['judge']['pass']for r in results if r['kind']=='real_subtle_edit'),
        'unchanged_strict_semantic_rubric':True,'new_evidence':'aligned RGB delta, fixed x4 difference crop'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('JUDGE_DELTA_CONTROLS',json.dumps(summary),flush=True)


if __name__=='__main__':main()
