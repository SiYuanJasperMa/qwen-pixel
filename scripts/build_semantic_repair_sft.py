"""Actual semantic-error histories, supervising only correct source-crop/RGB repairs."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFilter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import assistant_call,copy_source_image,execute_action,validate_outputs
from pixel.clean_edit_protocol import initial,program,TOOLS
from pixel.roi_augmentation import normalized_box,padded_box,recrop_polygons
from pixel.image_observations import load_observation


def raster(polygons,box,size):
    image=Image.new('L',size,0);draw=ImageDraw.Draw(image);x0,y0,x1,y1=box
    for polygon in polygons:draw.polygon([(x0+u*(x1-x0),y0+v*(y1-y0))for u,v in polygon],fill=255)
    return np.asarray(image)>0


def preserve_original_clip(code,box):
    """Recropping must not enlarge the original alpha's allowed source region."""
    if len(box)!=4 or any(not isinstance(v,int)for v in box)or box[2]<=box[0]or box[3]<=box[1]:
        raise ValueError('Need the actual original integer source box.')
    original='inside_region_np[y0:y1, x0:x1] = True'
    if code.count(original)!=1:raise ValueError('Unknown clipping implementation; do not silently change references.')
    x0,y0,x1,y1=box
    return code.replace(original,f'inside_region_np[{y0}:{y1}, {x0}:{x1}] = True')


def clip_partial_polygons(polygons,axis,cut,keep_low):
    """Teacher construction only: retain a genuine subset of a target contour."""
    result=[]
    for polygon in polygons:
        if not polygon:continue
        clipped=[]
        def inside(point):return point[axis]<=cut if keep_low else point[axis]>=cut
        previous=polygon[-1]
        for current in polygon:
            a,b=inside(previous),inside(current)
            if a!=b:
                scale=(cut-previous[axis])/(current[axis]-previous[axis])
                point=[previous[i]+scale*(current[i]-previous[i])for i in (0,1)]
                point[axis]=cut;clipped.append(point)
            if b:clipped.append(list(current))
            previous=current
        if len(clipped)>=3:result.append(clipped)
    return result


def build(task,data,out,fault,frozen_rows=None,repair_crop='padded'):
    reference=data/'executions'/task['id']
    if frozen_rows is not None:
        entries=[r for r in frozen_rows if r['meta']['trajectory_id']==task['id']]
        python_row=next(r for r in entries if r['target_message']['tool_calls'][0]['function']['name']=='python')
        crop_row=next(r for r in entries if r['target_message']['tool_calls'][0]['function']['name']=='view_image'and
                      r['target_message']['tool_calls'][0]['function']['arguments'].get('path')=='source.png')
        finish_row=next(r for r in entries if r['target_message']['tool_calls'][0]['function']['name']=='finish')
        correct=python_row['target_message']['tool_calls'][0]['function']['arguments']['code']
        source_block=next(b for m in python_row['messages']if m['role']=='user'for b in m['content']if b['type']=='image')
        source_image=load_observation(source_block);size=source_image.size;source=np.asarray(source_image)
        metadata=next(json.loads(b['text'])for m in reversed(python_row['messages'])if m['role']=='tool'
                      and isinstance(m['content'],list)for b in m['content']if b['type']=='text')
        box=metadata['source_bbox_pixels']
        output_block=next(b for m in reversed(finish_row['messages'])if isinstance(m.get('content'),list)
                          for b in m['content']if b['type']=='image')
        expected_rgb=np.asarray(load_observation(output_block))
        task={**task,'instruction':next(m['edit_instruction']for m in python_row['messages']if m['role']=='user')}
        first_crop=crop_row['target_message']['tool_calls'][0]['function']['arguments']
    else:
        trajectory=json.loads((reference/'trajectory.json').read_text())
        correct=trajectory['messages'][4]['tool_calls'][0]['function']['arguments']['code']
        box=json.loads((reference/'region.json').read_text())['source_bbox_pixels']
        with Image.open(reference/'source.png')as image:source_image=image.convert('RGB');size=image.size;source=np.asarray(source_image)
        expected_rgb=np.load(reference/'edited_rgb.npy',allow_pickle=False)
        first_crop={'path':'source.png','bbox_2d':normalized_box(box,size)}
    tree=ast.parse(correct)
    polygons=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='polygons_roi'for t in n.targets))
    mask=raster(polygons,box,size)
    if fault=='pure_partial':
        allowed=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
        candidates=[]
        for axis in [0,1]:
            for cut in [.25,.5,.75]:
                for low in [False,True]:
                    candidate=clip_partial_polygons(polygons,axis,cut,low)
                    candidate_mask=raster(candidate,box,size)
                    covered=np.count_nonzero(candidate_mask&mask)/np.count_nonzero(mask)
                    if .05<=covered<=.35 and not np.any(candidate_mask&~allowed):
                        candidates.append((abs(covered-.2),candidate))
        if not candidates:return [],{'id':task['id'],'fault':fault,'skip':'no clean omission contour'}
        wrong=min(candidates,key=lambda item:item[0])[1]
    else:
        wrong=([[[.35,0],[.65,0],[.65,.25],[.35,.25]]]if fault=='partial'else[[[0,0],[1,0],[1,1],[0,1]]])
    wrong_mask=raster(wrong,box,size)
    coverage=float(np.count_nonzero(mask&wrong_mask)/max(1,np.count_nonzero(mask)))
    spill=int(np.count_nonzero(wrong_mask&~mask))
    if fault in('partial','pure_partial')and coverage>.35:return [],{'id':task['id'],'fault':fault,'skip':'insufficient omission'}
    if fault=='spill'and spill<.2*np.count_nonzero(wrong_mask):return [],{'id':task['id'],'fault':fault,'skip':'insufficient unrelated area'}
    region=box if repair_crop=='original'else padded_box(box,size,.08)
    mapped=polygons if repair_crop=='original'else recrop_polygons(polygons,box,region)
    if any(not 0<=c<=1 for polygon in mapped for point in polygon for c in point):raise ValueError('ROI fractions violate the trained coordinate contract.')
    if not np.array_equal(mask,raster(mapped,region,size)):raise ValueError('Recropping changed the actual full-source mask.')
    identity=task['id']+'_repair_'+fault;workspace=out/'executions'/identity;workspace.mkdir(parents=True,exist_ok=True)
    source_image.save(workspace/'source.png');messages=initial(workspace/'source.png',task['instruction'])
    repaired_code=correct if repair_crop=='original'else preserve_original_clip(program(mapped,task['operation'],task.get('tint_color')or'red'),box)
    sequence=[assistant_call('view_image',first_crop),
        assistant_call('python',{'code':program(wrong,task['operation'],task.get('tint_color')or'red')}),
        assistant_call('view_image',{'path':'edited.png'}),
        assistant_call('view_image',{'path':'source.png','bbox_2d':normalized_box(region,size)}),
        assistant_call('python',{'code':repaired_code}),
        assistant_call('view_image',{'path':'edited.png'}),
        assistant_call('finish',{'image_path':'edited.png','rgb_path':'edited_rgb.npy'})]
    rows=[];bad_difference=None
    for step,action in enumerate(sequence):
        function=action['tool_calls'][0]['function']
        prefix=json.loads(json.dumps(messages))
        for attempt in range(3):
            feedback,passed=execute_action(function['name'],function['arguments'],workspace,step,True,True,True)
            if passed or 'TimeoutExpired'not in str(feedback):break
            with (workspace/'construction_retries.jsonl').open('a')as log:
                log.write(json.dumps({'step':step,'attempt':attempt,'feedback':feedback,
                    'same_annotation_action':True,'model_visible':False})+'\n')
        if not passed:raise ValueError('Repair reference execution failed: '+identity+' '+str(feedback))
        if step==1:
            candidates=workspace/'candidates';candidates.mkdir(exist_ok=True)
            shutil.copyfile(workspace/'edited.png',candidates/'bad.png');shutil.copyfile(workspace/'edited_rgb.npy',candidates/'bad_rgb.npy')
            good=expected_rgb.astype(float);bad=np.load(workspace/'edited_rgb.npy',allow_pickle=False).astype(float)
            baseline=np.abs(good-source).sum();bad_difference=float(np.abs(good-bad).sum()/max(1,baseline))
            if bad_difference<.2:return [],{'id':task['id'],'fault':fault,'skip':'candidate too similar to accepted result','relative_difference':bad_difference}
            if fault=='pure_partial':
                values=np.load(workspace/'edited_rgb.npy',allow_pickle=False).astype(np.int16)
                changed=np.any(np.abs(values-source.astype(np.int16))>1,axis=-1)
                outside_changed=int(np.count_nonzero(changed&~allowed))
                if outside_changed:return [],{'id':task['id'],'fault':fault,'skip':'omission also changes unrelated pixels','outside_gt_boundary2':outside_changed}
        if step>=3:
            rows.append({'id':identity+f'_step_{step}','task':'agent_edit','messages':prefix,
                'target_message':action,'tools':TOOLS,'meta':{'trajectory_id':identity,
                    'original_scene_id':task['original_scene_id'],'reference_id':task['id'],
                    'refinement':True,'supervised_actions_exclude_mistake':True,
                    'source':'Executed synthetic semantic mistake followed by annotation-derived full-mask repair',
                    'fault':fault}})
        messages.extend([action,feedback])
    receipt=validate_outputs(workspace)
    if not np.array_equal(np.load(workspace/'edited_rgb.npy',allow_pickle=False),expected_rgb):
        raise ValueError('Repair changed the verified native reference RGB: '+identity)
    record={'id':identity,'reference_id':task['id'],'image_id':task['original_scene_id'],'fault':fault,
        'instruction':task['instruction'],'source_image':str(workspace/'source.png'),
        'target_bbox_pixels':box,
        'bad_target_coverage':coverage,'bad_spill_pixels':spill,'bad_relative_rgb_difference':bad_difference,
        'correct_source_mask_exact':True,'actual_reference_repair_executed':True,
        'correct_reference_rgb_exact':True,'immutable_approved_reference':frozen_rows is not None,
        'repair_crop':repair_crop,
        'pure_omission_outside_changed_pixels_gt1_boundary2':0 if fault=='pure_partial'else None,
        'supervised_steps':[3,4,5,6],'bad_steps_unsupervised':[0,1,2],
        'output_image_sha256':receipt['image_sha256']}
    (workspace/'trajectory.json').write_text(json.dumps({'instruction':task['instruction'],'messages':messages,'repair_audit':record},indent=2))
    print('SEMANTIC_REPAIR_READY',json.dumps(record),flush=True);return rows,record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--images',type=int,default=24)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--faults',default='partial,spill')
    parser.add_argument('--approved-rows',type=Path,help='Use immutable current approved rows instead of mutable execution references')
    parser.add_argument('--repair-crop',choices=['original','padded'],default='padded')
    parser.add_argument('--balance-operations',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'train.jsonl').exists():raise ValueError('Use a new repair output variant.')
    approval_path=args.approved_rows or(args.data/'train_verified.jsonl')
    rows=[json.loads(l)for l in approval_path.open()]
    approved={r['meta']['trajectory_id']for r in rows};selected=[];images=set()
    groups={}
    for task in(json.loads(l)for l in(args.data/'tasks.jsonl').open()):
        if task['id']in approved:groups.setdefault(task['original_scene_id'],[]).append(task)
    for image_id,candidates in groups.items():
        desired=('blur','brighten','recolor')[len(selected)%3]
        task=next((t for t in candidates if t['operation']==desired),None)if args.balance_operations else candidates[0]
        if task is None:continue
        selected.append(task);images.add(image_id)
        if len(selected)==args.images:break
    if len(selected)!=args.images:raise ValueError('Not enough distinct verified training photos.')
    jobs=[(task,fault)for task in selected for fault in args.faults.split(',')]
    if any(fault not in ('partial','spill','pure_partial')for _,fault in jobs):raise ValueError('Unknown constructed semantic fault.')
    output=[];records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
        for built,record in pool.map(lambda job:build(job[0],args.data,args.out,job[1],rows if args.approved_rows else None,args.repair_crop),jobs):
            output.extend(built);records.append(record)
    (args.out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in output))
    originals={t['id']:t for t in selected}
    tasks=[{**originals[r['reference_id']],'id':r['id'],'reference_id':r['reference_id'],
            'instruction':r['instruction'],'image':r['source_image'],
            'target_bbox_pixels':r['target_bbox_pixels']}for r in records if 'skip'not in r]
    (args.out/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    manifest={'native_training_images':len({r['meta']['original_scene_id']for r in output}),
        'repair_trajectories':sum('skip'not in r for r in records),'assistant_actions':len(output),
        'source_data_sha256':hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        'source_masks_preserved_exactly':True,'all_feedback_actually_executed':True,
        'mistakes_supervised':False,'model_input_has_oracle_mask':False,
        'all_correct_rgb_exact':True,'immutable_approved_rows':bool(args.approved_rows),
        'repair_crop':args.repair_crop,
        'balanced_operations':args.balance_operations,
        'selected_operation_counts':{op:sum(t['operation']==op for t in selected)for op in ('blur','brighten','recolor')},
        'note':'Synthetic error histories with actual image feedback; correct repair is annotation-derived, not a claimed teacher model trajectory.',
        'records':records}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('SEMANTIC_REPAIR_DATA_READY',json.dumps({k:v for k,v in manifest.items()if k!='records'}),flush=True)


if __name__=='__main__':main()
