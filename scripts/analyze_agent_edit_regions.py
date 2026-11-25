"""Replay actual programs and measure alpha support against instance annotations.

These are spatial diagnostics, not official editing scores or semantic passes.
The annotation masks are never supplied to the actor or added to SFT inputs.
"""
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import execute_action,initial_messages,validate_outputs
from pixel.image_observations import load_observation
from scripts.build_clean_mask_sft import CATEGORIES


def instance_annotations(tasks):
    wanted={int(t['original_scene_id']):t for t in tasks}
    found={t['id']:{} for t in tasks}
    raw=ROOT.parent/'qwen_image_edit_smoke/data/public_raw/jxu124_refcoco'
    columns=['image_id','ann_id','raw_anns','raw_image_info','category_id']
    for batch in pq.ParquetFile(next(raw.glob('train-*.parquet'))).iter_batches(batch_size=512,columns=columns):
        for row in batch.to_pylist():
            task=wanted.get(row['image_id'])
            if task is None or CATEGORIES[row['category_id']]!=task['category']:continue
            annotation=json.loads(row['raw_anns']);info=json.loads(row['raw_image_info'])
            x,y,w,h=annotation['bbox']
            box=[max(0,math.floor(x)),max(0,math.floor(y)),min(info['width'],math.ceil(x+w)),min(info['height'],math.ceil(y+h))]
            if box!=task['target_bbox_pixels']:continue
            found[task['id']][row['ann_id']]={'annotation_id':row['ann_id'],'annotation':annotation,'image_info':info}
    result={}
    for task in tasks:
        candidates=list(found[task['id']].values())
        if len(candidates)!=1:raise ValueError('Expected one exact instance for '+task['id']+', got '+str(len(candidates)))
        result[task['id']]=candidates[0]
    return result


def raster_instance(record,size):
    segmentation=record['annotation']['segmentation']
    if not isinstance(segmentation,list):raise ValueError('Expected explicit instance polygons.')
    mask=Image.new('L',size,0);draw=ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon)>=6:draw.polygon([tuple(p)for p in np.asarray(polygon).reshape(-1,2)],fill=255)
    return np.asarray(mask)>0


def region_metrics(alpha,reference):
    if alpha.shape!=reference.shape or not np.isfinite(alpha).all()or alpha.min()<0 or alpha.max()>1:
        raise ValueError('Expected finite native-shape alpha in0-1.')
    peak=float(alpha.max());normalized=alpha/peak if peak else alpha
    support=normalized>=.5;low_support=normalized>.05
    intersection=int((support&reference).sum());union=int((support|reference).sum())
    mass=float(normalized.sum())
    return {'reference_pixels':int(reference.sum()),'support_pixels':int(support.sum()),
        'support_iou':intersection/union if union else None,
        'target_support_coverage':intersection/int(reference.sum()) if reference.any()else None,
        'target_low_alpha_coverage':float(low_support[reference].mean()) if reference.any()else None,
        'outside_target_support_pixels':int((support&~reference).sum()),
        'outside_target_alpha_mass_fraction':float(normalized[~reference].sum()/mass)if mass else None,
        'raw_alpha_peak':peak,'raw_alpha_mean_on_target':float(alpha[reference].mean())if reference.any()else None,
        'definition':'alpha normalized by its own peak for spatial support; raw opacity reported separately'}


def alpha_is_used_at_final_emit(code):
    tree=ast.parse(code)
    if not tree.body:return False
    final=tree.body[-1]
    if not(isinstance(final,ast.Expr)and isinstance(final.value,ast.Call)
           and isinstance(final.value.func,ast.Name)and final.value.func.id=='emit_rgb'
           and final.value.args):return False
    environment={}
    def dependencies(expression):
        result=set()
        for node in ast.walk(expression):
            if isinstance(node,ast.Name)and isinstance(node.ctx,ast.Load):
                result.update(environment.get(node.id,{node.id}))
        return result
    for statement in tree.body[:-1]:
        if isinstance(statement,ast.Assign):
            values=dependencies(statement.value)
            for target in statement.targets:
                if isinstance(target,ast.Name):environment[target.id]=values|{target.id}
        elif isinstance(statement,ast.AnnAssign)and isinstance(statement.target,ast.Name)and statement.value is not None:
            environment[statement.target.id]=dependencies(statement.value)|{statement.target.id}
    return 'alpha_np'in dependencies(final.value.args[0])


def analyze(task,evaluation,annotation,folder):
    original=Path(evaluation['artifact_dir']);trace=json.loads((original/'trajectory.json').read_text())
    python_turns=[t for t in trace['turns']if t.get('action')=='python']
    if len(python_turns)!=1 or not python_turns[0].get('passed'):
        raise ValueError('This isolated-mask diagnostic requires one successful Python action.')
    step=python_turns[0]['step']
    region_turns=[t for t in trace['turns']if t['step']<step and t.get('action')=='view_image'and t.get('passed')and t['arguments'].get('bbox_2d')is not None]
    if len(region_turns)!=1:raise ValueError('Expected one actual source-region observation.')
    block=next(b for m in trace['messages']if m['role']=='user'and isinstance(m['content'],list)for b in m['content']if b.get('type')=='image')
    image=load_observation(block)
    folder.mkdir(parents=True,exist_ok=False);image.save(folder/'source.png')
    initial_messages(folder/'source.png','isolated spatial diagnostic')
    _feedback,passed=execute_action('view_image',region_turns[0]['arguments'],folder,0,True,True,True)
    if not passed:raise ValueError('The actual source-region replay failed.')
    code=python_turns[0]['arguments']['code']
    if not alpha_is_used_at_final_emit(code):raise ValueError('Cannot verify alpha_np contributes to the final emitted RGB.')
    code+='\n_diagnostic_alpha_np = np.asarray(alpha_np)\nnp.save("diagnostic_alpha.npy", _diagnostic_alpha_np)\n'
    feedback,passed=execute_action('python',{'code':code},folder,1,True,True,True)
    if not passed:raise ValueError('Program diagnostic replay failed: '+json.dumps(feedback))
    validate_outputs(folder)
    original_rgb=np.load(original/evaluation.get('validation',{}).get('rgb_path','edited_rgb.npy'),allow_pickle=False)
    replay_rgb=np.load(folder/'edited_rgb.npy',allow_pickle=False)
    if not np.array_equal(original_rgb,replay_rgb):raise ValueError('Isolated replay differs from the actual submitted RGB.')
    alpha=np.load(folder/'diagnostic_alpha.npy',allow_pickle=False).astype(np.float32)
    info=annotation['image_info']
    if image.size!=(info['width'],info['height']):raise ValueError('Source and instance dimensions differ.')
    reference=raster_instance(annotation,image.size)
    metrics=region_metrics(alpha,reference)
    Image.fromarray(reference.astype(np.uint8)*255).save(folder/'reference_mask.png')
    peak=float(alpha.max());Image.fromarray(np.rint((alpha/peak if peak else alpha)*255).astype(np.uint8)).save(folder/'predicted_alpha.png')
    source=np.asarray(image).astype(np.float32);overlay=source.copy()
    selected=(alpha/peak if peak else alpha)>=.5
    overlay[selected]=.6*overlay[selected]+.4*np.array([255,20,20])
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(folder/'predicted_overlay.png')
    return {'id':task['id'],'semantic_pass':evaluation.get('judge',{}).get('pass'),
        'exact_actual_rgb_replay':True,'annotation_id':annotation['annotation_id'],
        'annotation_sha256':hashlib.sha256(json.dumps(annotation,sort_keys=True).encode()).hexdigest(),
        'metrics':metrics,'workspace':str(folder),'diagnostic_only':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--evaluations',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(s)for s in args.tasks.read_text().splitlines()if s.strip()]
    annotations=instance_annotations(tasks);results=[]
    for task in tasks:
        path=args.evaluations/task['id']/'evaluation.json'
        if not path.exists():continue
        evaluation=json.loads(path.read_text())
        if not evaluation.get('complete'):continue
        try:result=analyze(task,evaluation,annotations[task['id']],args.out/task['id'])
        except Exception as error:result={'id':task['id'],'diagnostic_error':type(error).__name__+': '+str(error)}
        results.append(result);(args.out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'observed':len(results),'measured':sum('metrics'in r for r in results),
        'diagnostic_only':True,'not_official_or_semantic_score':True},indent=2))


if __name__=='__main__':main()
