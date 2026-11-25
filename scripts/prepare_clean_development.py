"""Freeze image-disjoint development edits without adding supervision to SFT."""
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from build_clean_mask_sft import CATEGORIES,prepare_mask,ROOT


def main():
    data=ROOT/'data/sft_clean_mask_v2'
    training=[json.loads(l)for l in(data/'tasks.jsonl').open()]
    excluded={int(t['original_scene_id'])for t in training}
    excluded|={int(t['original_scene_id'])for t in(json.loads(l)for l in(ROOT/'data/agent_native_v4/tasks.jsonl').open())if t['split']=='val'}
    excluded|={int(t['image_id'])for t in(json.loads(l)for l in(ROOT/'data/independent_coco_edit_eval_v2/tasks.jsonl').open())}
    raw=ROOT.parent/'qwen_image_edit_smoke/data/public_raw/jxu124_refcoco'
    photos=ROOT.parent/'qwen_image_edit_smoke/data/refcoco_train_real_scale_v1/originals'
    candidates=[];seen=set()
    columns=['image_id','ann_id','split','raw_anns','raw_image_info','captions','category_id']
    for batch in pq.ParquetFile(next(raw.glob('train-*.parquet'))).iter_batches(batch_size=256,columns=columns):
        for ref in batch.to_pylist():
            if ref['split']!='train'or ref['image_id']in excluded or ref['ann_id']in seen:continue
            seen.add(ref['ann_id']);source=photos/f'COCO_train2014_{ref["image_id"]:012d}.jpg'
            if not source.exists()or not ref['captions']:continue
            annotation=json.loads(ref['raw_anns']);info=json.loads(ref['raw_image_info'])
            if annotation.get('iscrowd')or not isinstance(annotation.get('segmentation'),list):continue
            x,y,w,h=annotation['bbox']
            if w<40 or h<40 or not .04<=annotation['area']/(info['width']*info['height'])<=.65:continue
            order=hashlib.sha256(f'clean-dev-v1:{ref["image_id"]}:{ref["ann_id"]}'.encode()).hexdigest()
            candidates.append((order,ref,annotation,info,source))
    candidates.sort(key=lambda t:t[0]);tasks=[];used=set();counts={}
    for _,ref,annotation,info,source in candidates:
        category=ref['category_id']
        if ref['image_id']in used or counts.get(category,0)>=2:continue
        x,y,w,h=annotation['bbox'];size=(info['width'],info['height'])
        box=[max(0,math.floor(x)),max(0,math.floor(y)),min(size[0],math.ceil(x+w)),min(size[1],math.ceil(y+h))]
        prepared=prepare_mask(annotation,size,box)
        if prepared is None:continue
        operation=('blur','brighten','recolor')[len(tasks)%3]
        if operation=='recolor'and category==1:operation='brighten'
        color=('red','green','blue','purple')[ref['image_id']%4]
        target=f'the {CATEGORIES[category]} described as "{ref["captions"][0]}"'
        instruction=(f'Blur the entire visible area of {target}. Keep the rest of the image unchanged.'if operation=='blur'else
            f'Moderately brighten the entire visible area of {target}, retaining details and keeping the rest unchanged.'if operation=='brighten'else
            f'Apply a subtle {color} tint to the entire visible surface of {target}, retaining texture and shape and keeping the rest unchanged.')
        tasks.append({'id':f'clean_development_{len(tasks):02d}','image':str(source),'instruction':instruction,
            'operation':operation,'category':CATEGORIES[category],'original_scene_id':str(ref['image_id']),
            'split':'val','strict_target_extent':True,'evaluation_only':True,'target_bbox_pixels':box})
        used.add(ref['image_id']);counts[category]=counts.get(category,0)+1
        if len(tasks)==8:break
    if len(tasks)!=8:raise ValueError('Unable to construct disjoint development edits.')
    text=''.join(json.dumps(t)+'\n'for t in tasks);(data/'development_tasks.jsonl').write_text(text)
    (data/'development_manifest.json').write_text(json.dumps({'images':8,'training_image_overlap':0,
        'evaluation_only':True,'strict_extent':True,'tasks_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'selection':'Frozen annotation-based source selection before model scoring'},indent=2))
    print('CLEAN_DEVELOPMENT_FROZEN',[(t['category'],t['operation'])for t in tasks],flush=True)


if __name__=='__main__':main()
