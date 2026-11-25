"""Review executed reference edits and retain complete, strictly accepted trajectories."""
import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import validate_outputs
from pixel.teacher_api import make_client,judge_edit


def review(task,data,out):
    folder=data/'executions'/task['id']
    receipt=validate_outputs(folder)
    identity={'image_sha256':receipt['image_sha256'],
        'source_sha256':hashlib.sha256((folder/'source.png').read_bytes()).hexdigest(),
        'instruction':task['instruction'],'strict_extent':True,
        'judge_code_sha256':hashlib.sha256((ROOT/'pixel/teacher_api.py').read_bytes()).hexdigest(),
        'difference_code_sha256':hashlib.sha256((ROOT/'pixel/edit_difference.py').read_bytes()).hexdigest()}
    path=out/(task['id']+'.json')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('identity')!=identity:raise ValueError('Stale reference review: '+task['id'])
        return saved
    box=task.get('target_bbox_pixels')or json.loads((folder/'region.json').read_text())['source_bbox_pixels']
    judgment=judge_edit(make_client(),folder/'source.png',folder/'edited.png',task['instruction'],True,box)
    result={'id':task['id'],'operation':task['operation'],'judge':judgment,
            'identity':identity,'new_teacher_generation':False}
    path.write_text(json.dumps(result,indent=2));return result


def review_pair(task,data,out):
    good=review(task,data,out)
    folder=data/'executions'/task['id'];bad=folder/'candidates/bad.png'
    if not bad.exists():raise ValueError('Paired repair review requires the actual saved bad candidate.')
    identity={**good['identity'],'bad_image_sha256':hashlib.sha256(bad.read_bytes()).hexdigest()}
    path=out/(task['id']+'_bad.json')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('Saved bad candidate review no longer matches its pixels.')
        judgment=saved['judge']
    else:
        box=task.get('target_bbox_pixels')or json.loads((folder/'region.json').read_text())['source_bbox_pixels']
        judgment=judge_edit(make_client(),folder/'source.png',bad,task['instruction'],True,box)
        path.write_text(json.dumps({'id':task['id'],'judge':judgment,'identity':identity},indent=2))
    return {**good,'bad_judge':judgment}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/sft_clean_mask_v2')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--write-verified',action='store_true')
    parser.add_argument('--mask-reviews',type=Path)
    parser.add_argument('--require-bad-fail',action='store_true')
    args=parser.parse_args()
    data=args.data.resolve();out=args.out or ROOT/'outputs'/data.name/'reference_review_boxed'
    out.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(l)for l in(data/'tasks.jsonl').open()]
    if args.mask_reviews:
        approved={}
        for path in args.mask_reviews.glob('*.json'):
            mask_review=json.loads(path.read_text())
            if mask_review.get('accepted')is True:
                approved[mask_review['source_task'].rsplit('_',1)[0]]=mask_review
        tasks=[t for t in tasks if 'clean_'+t['original_scene_id']+'_'+t['id'].split('_')[2]in approved]
        for task in tasks:
            mask_review=approved[task['id'].rsplit('_',2)[0]if task['operation']=='recolor'else task['id'].rsplit('_',1)[0]]
            source=data/'executions'/task['id']/'source.png'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=mask_review['source_sha256']:
                raise ValueError('Reviewed mask source image changed.')
    selected=tasks if args.all else [t for operation in ('blur','brighten','recolor')
        for t in [t for t in tasks if t['operation']==operation][:2]]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)as pool:
        reviewer=review_pair if args.require_bad_fail else review
        pending={pool.submit(reviewer,t,data,out):t for t in selected}
        for future in concurrent.futures.as_completed(pending):
            task=pending[future]
            try:result=future.result()
            except Exception as error:result={'id':task['id'],'operation':task['operation'],'review_error':type(error).__name__}
            results.append(result);print(json.dumps(result),flush=True)
    accepted={r['id']for r in results if r.get('judge',{}).get('pass')is True
              and(not args.require_bad_fail or r.get('bad_judge',{}).get('pass')is False)}
    summary={'references_reviewed':len(results),'strict_pass':len(accepted),
        'review_errors':sum('review_error'in r for r in results),
        'operations':{op:{'reviewed':sum(r['operation']==op for r in results),
            'accepted':sum(r['operation']==op and r['id']in accepted for r in results)}for op in ('blur','brighten','recolor')},
        'note':'Reference quality only; not evidence of Qwen generalization.'}
    summary['mask_approved_reference_candidates']=len(tasks)if args.mask_reviews else None
    summary['require_actual_bad_image_to_fail']=args.require_bad_fail
    summary['constructed_bad_candidates_judged_acceptable']=sum(r.get('bad_judge',{}).get('pass')is True for r in results)
    if args.write_verified:
        if not args.all or summary['review_errors']:raise ValueError('Verify all references successfully before filtering.')
        rows=[json.loads(l)for l in(data/'train.jsonl').open()]
        verified=[r for r in rows if r['meta']['trajectory_id']in accepted]
        if len(verified)!=4*len(accepted):raise ValueError('Incomplete accepted training trajectories.')
        (data/'train_verified.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in verified))
        summary['verified_assistant_actions']=len(verified)
        summary['verified_native_images']=len({r['meta']['original_scene_id']for r in verified})
        summary['verified_data_sha256']=hashlib.sha256((data/'train_verified.jsonl').read_bytes()).hexdigest()
        (data/'reference_review_manifest.json').write_text(json.dumps(summary,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('REFERENCE_REVIEW',json.dumps(summary),flush=True)


if __name__=='__main__':main()
