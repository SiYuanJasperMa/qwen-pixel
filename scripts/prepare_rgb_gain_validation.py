"""Freeze balanced held-out tasks for paired image versus image+RGB tests."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.rgb_text import encode_rgb,decode_rgb
from pixel.data_lineage import source_group
from pixel.persistent_io import atomic_text


def read(path):return[json.loads(l)for l in path.read_text().splitlines()if l.strip()]


def main():
    out=ROOT/'data/gpt61_rgb_gain_val64_v1';out.mkdir(exist_ok=True)
    pools={}
    pools['magicbrush']=read(ROOT/'data/magicbrush_official_dev_v1/tasks.jsonl')
    old=read(ROOT/'data/star_clean_v2/tasks_val.jsonl')
    pools['ultraedit']=[r for r in old if r['dataset']=='ultraedit']
    pools['sciforma']=[r for r in old if r['dataset']=='sciforma']
    new=read(ROOT/'data/editing_growth_v1/tasks_val.jsonl')
    pools['maskedit_en']=[r for r in new if r['dataset']=='maskedit_en']
    pools['maskedit_cn']=[r for r in new if r['dataset']=='maskedit_cn']
    target={'magicbrush':16,'ultraedit':16,'sciforma':16,'maskedit_en':8,'maskedit_cn':8}
    train_groups=set()
    for fn in ['data/star_clean_v2/sft_mixed_aug/train.jsonl','data/editing_growth_v1/sft/train.jsonl','data/editing_growth_v1/gpt62/sft/train.jsonl']:
        train_groups.update(r['meta']['source_group_id']for r in read(ROOT/fn))
    prior={r['id']for r in read(ROOT/'data/rgb_text_smoke_v1/tasks.jsonl')}
    selected=[];groups=set()
    for ds,pool in pools.items():
        random.Random(20261124).shuffle(pool)
        chosen=0
        for task in pool:
            if task['id']in prior:continue
            image=Path(task['image']);image=image if image.is_absolute()else ROOT/image
            group=task.get('source_group_id')or source_group(task['id'],image)
            if group in groups or group in train_groups:continue
            selected.append({**task,'dataset':ds,'image':str(image),'source_group_id':group})
            groups.add(group);chosen+=1
            if chosen==target[ds]:break
        if chosen!=target[ds]:raise ValueError(f'Not enough isolated {ds} groups: {chosen}')
    assert len(selected)==64
    prepared=[]
    for i,task in enumerate(selected):
        folder=out/task['id'];folder.mkdir(exist_ok=True)
        source=folder/'source128.png';target_image=folder/'target128.png'
        for src,dst in [(task['image'],source),(task['target_image'],target_image)]:
            p=Path(src);p=p if p.is_absolute()else ROOT/p
            with Image.open(p)as im:im.convert('RGB').resize((128,128),Image.Resampling.LANCZOS).save(dst)
        with Image.open(source)as im:
            rgb=encode_rgb(im);assert np.array_equal(np.asarray(im.convert('RGB')),np.asarray(decode_rgb(rgb)))
        path=folder/'source.rgb.txt';atomic_text(path,rgb)
        prepared.append({'id':task['id'],'dataset':task['dataset'],'instruction':task['instruction'],'image':str(source),
                         'target_image':str(target_image),'rgb_text':str(path),'width':128,'height':128,
                         'source_group_id':task['source_group_id'],'split':'val','evaluation_only':True,
                         'source_rgb_sha256':hashlib.sha256(decode_rgb(rgb).tobytes()).hexdigest()})
        atomic_text(out/'tasks.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in prepared))
        if i%8==7:print('PREPARED',i+1,64,flush=True)
    atomic_text(out/'manifest.json',json.dumps({'tasks':64,'datasets':dict(Counter(r['dataset']for r in prepared)),
               'unique_source_groups':len(groups),'train_source_group_overlap':0,'resolution':'128x128 explicitly resized',
               'modes':['image_only','image_rgb'],'teacher':'traex/GPT-6.1-Sol','reasoning_effort':'high','max_output_tokens':8192,
               'evaluation_only':True,'sampling_seed':20261124,'prior_three_smoke_excluded':True},indent=2))


if __name__=='__main__':main()
