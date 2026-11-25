"""Diagnostic: do coordinate losses react to image content after SFT?

This intentionally inconsistent input is diagnostic only and never becomes SFT data.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode,load
from pixel.image_observations import load_observation


def image_blocks(row):
    return [b for m in row['messages']if isinstance(m['content'],list)
            for b in m['content']if b.get('type')=='image']


def source_identity(row):
    scene=row.get('meta',{}).get('original_scene_id')
    if scene is not None:return str(scene)
    return hashlib.sha256(load_observation(image_blocks(row)[0]).tobytes()).hexdigest()


def losses(model,batch):
    weights=batch.pop('token_weights')
    with torch.inference_mode():
        result=model(**batch,use_cache=False)
    labels=batch['labels'][:,1:];valid=labels!=-100;geometry=(weights[:,1:]>1)&valid
    ce=F.cross_entropy(result.logits[:,:-1].float().reshape(-1,result.logits.shape[-1]),
        labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(labels)
    return {'geometry_loss':float(ce[geometry].mean())if geometry.any()else None,
            'other_assistant_loss':float(ce[valid&~geometry].mean())if(valid&~geometry).any()else None,
            'geometry_tokens':int(geometry.sum()),'assistant_loss':float(ce[valid].mean())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',required=True)
    parser.add_argument('--data',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model')
    parser.add_argument('--images',type=int,default=6);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l)for l in Path(args.data).open()]
    selected=[];seen=set()
    for row in rows:
        calls=row.get('target_message',{}).get('tool_calls',[])
        identity=source_identity(row)
        if identity in seen or not calls or calls[0]['function']['name']!='python':continue
        selected.append(row);seen.add(identity)
        if len(selected)==args.images:break
    if len(selected)<2:raise ValueError('Visual-conditioning checks require distinct source trajectories.')
    from pixel.decoding import configure_decoding
    configure_decoding('deterministic')
    model_path=args.model or json.loads((Path(args.adapter)/'adapter_config.json').read_text())['base_model_name_or_path']
    model,processor=load(SimpleNamespace(model=model_path,
        adapter=args.adapter,mode='generate',image_max_pixels=262144))
    model.eval();results=[]
    for index,row in enumerate(selected):
        swapped=copy.deepcopy(row)
        candidates=image_blocks(selected[(index+1)%len(selected)])
        for image_index,block in enumerate(image_blocks(swapped)):
            size=load_observation(block).size
            wrong=load_observation(candidates[image_index%len(candidates)]).resize(size)
            path=args.out/f'swapped_{index}_{image_index}.png';wrong.save(path);block['image']=str(path)
            block['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        good=encode(processor,row,model.device,True,8192,1.,3.)
        wrong=encode(processor,swapped,model.device,True,8192,1.,3.)
        for key in ('input_ids','labels','attention_mask','mm_token_type_ids','image_grid_thw'):
            if key not in good:continue
            if not torch.equal(good[key],wrong[key]):raise ValueError('Content perturbation changed token/geometry alignment: '+key)
        a=losses(model,good);b=losses(model,wrong)
        result={'id':row['id'],'original':a,'different_image_same_geometry':b,
            'source_identity':source_identity(row),'donor_identity':source_identity(selected[(index+1)%len(selected)]),
            'geometry_loss_increase':b['geometry_loss']-a['geometry_loss']if a['geometry_loss']is not None else None,
            'assistant_loss_increase':b['assistant_loss']-a['assistant_loss'],
            'diagnostic_only':True}
        results.append(result);print(json.dumps(result),flush=True)
        (args.out/'results.json').write_text(json.dumps(results,indent=2))
    geometry_deltas=[r['geometry_loss_increase']for r in results if r['geometry_loss_increase']is not None]
    summary={'samples':len(results),'geometry_measured_samples':len(geometry_deltas),'mean_geometry_loss_increase':
        sum(geometry_deltas)/len(geometry_deltas)if geometry_deltas else None,
        'mean_assistant_loss_increase':sum(r['assistant_loss_increase']for r in results)/len(results),
        'note':'Image sensitivity diagnostic, not a semantic editing score or train data.'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print('VISUAL_CONDITIONING_AUDIT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
