"""Small genuine-photo SFT with correct instance masks and uniform RGB programs."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import assistant_call,copy_source_image,execute_action,validate_outputs
from pixel.clean_edit_protocol import initial,program,TOOLS

CATEGORIES={1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',11:'fire hydrant',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',27:'backpack',28:'umbrella',31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',67:'dining table',70:'toilet',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',81:'sink',82:'refrigerator',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush'}


def rdp(points,epsilon):
    if len(points)<3:return points
    a,b=points[0],points[-1];delta=b-a
    if np.dot(delta,delta)==0:dist=np.linalg.norm(points-a,axis=1)
    else:
        projection=np.clip(((points-a)@delta)/np.dot(delta,delta),0,1)
        dist=np.linalg.norm(points-(a+projection[:,None]*delta),axis=1)
    index=int(dist.argmax())
    if dist[index]<=epsilon:return np.vstack([a,b])
    return np.vstack([rdp(points[:index+1],epsilon)[:-1],rdp(points[index:],epsilon)])


def simplify(points,epsilon):
    points=np.asarray(points,dtype=float).reshape(-1,2)
    if np.array_equal(points[0],points[-1]):points=points[:-1]
    if len(points)<4:return points
    far=int(np.linalg.norm(points-points[0],axis=1).argmax())
    return np.vstack([rdp(points[:far+1],epsilon)[:-1],
                      rdp(np.vstack([points[far:],points[0]]),epsilon)[:-1]])


def raster(polygons,size):
    mask=Image.new('L',size,0);draw=ImageDraw.Draw(mask)
    for points in polygons:draw.polygon([tuple(p)for p in points],fill=255)
    return np.asarray(mask)>0


def prepare_mask(annotation,size,box):
    polygons=[np.asarray(p,dtype=float).reshape(-1,2)for p in annotation['segmentation']if len(p)>=6]
    if not polygons:return None
    original=raster(polygons,size)
    x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
    for epsilon in (3.,2.,1.,.5):
        simplified=[simplify(p,epsilon)for p in polygons]
        if any(len(p)<3 for p in simplified):continue
        fractions=[[[round((float(x)-x0)/bw,4),round((float(y)-y0)/bh,4)]for x,y in p]for p in simplified]
        recovered=[[(x0+u*bw,y0+v*bh)for u,v in p]for p in fractions]
        candidate=raster(recovered,size)
        intersection=np.count_nonzero(candidate&original);union=np.count_nonzero(candidate|original)
        coverage=intersection/max(1,np.count_nonzero(original));iou=intersection/max(1,union)
        vertices=sum(len(p)for p in fractions)
        if coverage>=.97 and iou>=.95 and vertices<=80:
            return fractions,{'mask_coverage':float(coverage),'mask_iou':float(iou),'vertices':vertices}
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--images',type=int,default=64)
    parser.add_argument('--variant',default='sft_clean_mask_v2')
    parser.add_argument('--mask-reviews',type=Path)
    parser.add_argument('--selection-manifest',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume-incomplete',action='store_true')
    args=parser.parse_args()
    out=ROOT/'data'/args.variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'train.jsonl').exists():
        raise ValueError('Use a fresh variant; do not overwrite executed observations.')
    approved={}
    if args.mask_reviews:
        for path in args.mask_reviews.glob('*.json'):
            review=json.loads(path.read_text())
            if review.get('accepted')is True:
                # Approval belongs to this exact annotated instance, not any object in its photo.
                approved[review['source_task'].rsplit('_',1)[0]]=review
    selection=None
    if args.selection_manifest:
        selection={int(q['image_id'])for q in json.loads(args.selection_manifest.read_text())['quality']}
    photos=ROOT.parent/'qwen_image_edit_smoke/data/refcoco_train_real_scale_v1/originals'
    raw=ROOT.parent/'qwen_image_edit_smoke/data/public_raw/jxu124_refcoco'
    excluded={int(t['original_scene_id'])for t in(json.loads(l)for l in(ROOT/'data/agent_native_v4/tasks.jsonl').open())if t['split']=='val'}
    excluded|={int(t['image_id'])for t in(json.loads(l)for l in(ROOT/'data/independent_coco_edit_eval_v2/tasks.jsonl').open())}
    development=ROOT/'data/sft_clean_mask_v2/development_tasks.jsonl'
    if development.exists():
        excluded|={int(t['original_scene_id'])for t in(json.loads(l)for l in development.open())}
    candidates=[];seen=set()
    columns=['image_id','ann_id','split','raw_anns','raw_image_info','captions','category_id']
    for batch in pq.ParquetFile(next(raw.glob('train-*.parquet'))).iter_batches(batch_size=256,columns=columns):
        for ref in batch.to_pylist():
            if ref['split']!='train'or ref['image_id']in excluded or ref['ann_id']in seen:continue
            if selection is not None and ref['image_id']not in selection:continue
            instance=f'clean_{ref["image_id"]}_{ref["ann_id"]}'
            if args.mask_reviews and instance not in approved:continue
            seen.add(ref['ann_id'])
            source=photos/f'COCO_train2014_{ref["image_id"]:012d}.jpg'
            if not source.exists()or not ref['captions']:continue
            annotation=json.loads(ref['raw_anns']);info=json.loads(ref['raw_image_info'])
            if annotation.get('iscrowd')or not isinstance(annotation.get('segmentation'),list):continue
            x,y,w,h=annotation['bbox'];fraction=annotation['area']/(info['width']*info['height'])
            if w<40 or h<40 or not .04<=fraction<=.65:continue
            order=hashlib.sha256(f'clean-sft-v1:{ref["image_id"]}:{ref["ann_id"]}'.encode()).hexdigest()
            candidates.append((order,ref,annotation,info,source))
    candidates.sort(key=lambda item:item[0])
    selected=[];used_images=set();category_counts={}
    for _,ref,annotation,info,source in candidates:
        category=ref['category_id'];limit=20 if category==1 else 8
        if ref['image_id']in used_images or category_counts.get(category,0)>=limit:continue
        size=(info['width'],info['height']);x,y,w,h=annotation['bbox']
        box=[max(0,math.floor(x)),max(0,math.floor(y)),min(size[0],math.ceil(x+w)),min(size[1],math.ceil(y+h))]
        mask=prepare_mask(annotation,size,box)
        if mask is None:continue
        polygons,quality=mask
        selected.append((ref,source,size,box,polygons,quality));used_images.add(ref['image_id'])
        category_counts[category]=category_counts.get(category,0)+1
        if len(selected)==args.images:break
    if len(selected)<args.images:raise ValueError('Not enough native masks meeting full-object fidelity checks.')
    def build_instance(item):
        ref,source,size,box,polygons,quality=item
        rows=[];tasks=[]
        with Image.open(source)as source_image:
            if source_image.size!=size:raise ValueError('Native source size differs from its annotation.')
        category=ref['category_id'];description=ref['captions'][0]
        operations=['blur','brighten']+([]if category==1 else['recolor'])
        for operation in operations:
            color=('red','green','blue','purple')[ref['image_id']%4]
            identity=f'clean_{ref["image_id"]}_{ref["ann_id"]}_{operation}'+('_'+color if operation=='recolor'else'')
            workspace=out/'executions'/identity;workspace.mkdir(parents=True,exist_ok=True)
            previous_source_sha=None
            if args.resume_incomplete and (workspace/'trajectory.json').exists():
                previous_source_sha=hashlib.sha256((workspace/'source.png').read_bytes()).hexdigest()
            copy_source_image(source,workspace/'source.png')
            if previous_source_sha is not None and hashlib.sha256((workspace/'source.png').read_bytes()).hexdigest()!=previous_source_sha:
                raise ValueError('Cached reference source pixels changed: '+identity)
            if args.mask_reviews:
                review=approved[f'clean_{ref["image_id"]}_{ref["ann_id"]}']
                if hashlib.sha256((workspace/'source.png').read_bytes()).hexdigest()!=review['source_sha256']:
                    raise ValueError('Reviewed source differs from rebuilt source.')
            if operation=='blur':instruction=f'Blur the entire visible {CATEGORIES[category]} described as "{description}". Keep the rest of the image unchanged.'
            elif operation=='brighten':instruction=f'Moderately brighten the entire visible {CATEGORIES[category]} described as "{description}", preserving details and keeping the rest unchanged.'
            else:instruction=f'Apply a subtle {color} tint to the entire visible surface of the {CATEGORIES[category]} described as "{description}", retaining texture and shape and keeping the rest unchanged.'
            messages=initial(workspace/'source.png',instruction)
            normalized=[round(value*1000/(size[0]if i%2==0 else size[1]))for i,value in enumerate(box)]
            mapped=[round(value*(size[0]if i%2==0 else size[1])/1000)for i,value in enumerate(normalized)]
            if mapped!=box:normalized=[round(value*1000/(size[0]if i%2==0 else size[1]),8)for i,value in enumerate(box)]
            sequence=[assistant_call('view_image',{'path':'source.png','bbox_2d':normalized}),
                      assistant_call('python',{'code':program(polygons,operation,color)}),
                      assistant_call('view_image',{'path':'edited.png'}),
                      assistant_call('finish',{'image_path':'edited.png','rgb_path':'edited_rgb.npy'})]
            trajectory_rows=[]
            cached=None
            if args.resume_incomplete and (workspace/'trajectory.json').exists():
                stored=json.loads((workspace/'trajectory.json').read_text())
                cached=stored['messages']
                if cached[:2]!=messages or len(cached)!=10:
                    raise ValueError('Cached reference input/sequence changed: '+identity)
                for step,action in enumerate(sequence):
                    if cached[2+step*2]!=action:raise ValueError('Cached reference code/action changed: '+identity)
                validate_outputs(workspace)
            for step,action in enumerate(sequence):
                call=action['tool_calls'][0]['function']
                if cached is not None:
                    feedback,passed=cached[3+step*2],True
                else:
                    for attempt in range(3):
                        feedback,passed=execute_action(call['name'],call['arguments'],workspace,step,True,True,True)
                        if passed or 'TimeoutExpired'not in str(feedback):break
                        # Infrastructure retries are external construction logs, never phantom model-visible tool turns.
                        with (workspace/'construction_retries.jsonl').open('a')as log:
                            log.write(json.dumps({'step':step,'attempt':attempt,'feedback':feedback,'same_reference_action':True})+'\n')
                if not passed:raise ValueError('Clean reference program failed: '+identity+' '+str(feedback))
                trajectory_rows.append({'id':identity+f'_step_{step}','task':'agent_edit',
                    'messages':json.loads(json.dumps(messages)),'target_message':action,'tools':TOOLS,
                    'meta':{'trajectory_id':identity,'original_scene_id':str(ref['image_id']),
                            'source':'COCO_instance_annotation_program_supervision; not GPT hidden reasoning',**quality}})
                messages.extend([action,feedback])
            rows.extend(trajectory_rows)
            (workspace/'trajectory.json').write_text(json.dumps({'instruction':instruction,'messages':messages,
                'annotation_mask_quality':quality,'reference_program_executed':True},indent=2))
            tasks.append({'id':identity,'image':str(source),'instruction':instruction,'operation':operation,
                'tint_color':color if operation=='recolor'else None,
                'original_scene_id':str(ref['image_id']),'scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'split':'train'})
        print('CLEAN_INSTANCE',ref['image_id'],CATEGORIES[category],quality,flush=True)
        return rows,tasks,{'image_id':ref['image_id'],'category':CATEGORIES[category],**quality}
    rows=[];tasks=[];quality_rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
        for scene_rows,scene_tasks,quality in pool.map(build_instance,selected):
            rows.extend(scene_rows);tasks.extend(scene_tasks);quality_rows.append(quality)
    (out/'train.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
    (out/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    manifest={'native_training_images':len(selected),'editing_trajectories':len(tasks),'assistant_actions':len(rows),
        'category_counts':{CATEGORIES[k]:v for k,v in category_counts.items()},
        'actual_reference_programs_executed':True,'mask_coverage_min':min(q['mask_coverage']for q in quality_rows),
        'mask_iou_min':min(q['mask_iou']for q in quality_rows),'heldout_image_overlap':0,
        'mask_polygons_in_model_input':False,'source':'Existing native RefCOCO train photos and raw COCO instance annotations',
        'program_sha256':hashlib.sha256(Path(ROOT/'pixel/clean_edit_protocol.py').read_bytes()).hexdigest(),
        'mask_reviews':str(args.mask_reviews)if args.mask_reviews else None,
        'construction_workers':args.workers,'resumed_incomplete_construction':args.resume_incomplete,
        'rgb_transforms':{'brighten':'gamma=0.75','tint':'25% max-value tint, 75% original'},
        'quality':quality_rows}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CLEAN_MASK_SFT_READY',json.dumps({k:v for k,v in manifest.items()if k!='quality'}),flush=True)


if __name__=='__main__':main()
