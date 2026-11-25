"""Measure full generated polygons on real training prefixes; no gold target tokens."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode,load
from pixel.agent_protocol import parse_call


def polygons(code):
    tree=ast.parse(code)
    return next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='polygons_roi'for t in n.targets))


def mask(points,box,size):
    x0,y0,x1,y1=box;image=Image.new('L',size,0);draw=ImageDraw.Draw(image)
    for polygon in points:draw.polygon([(x0+u*(x1-x0),y0+v*(y1-y0))for u,v in polygon],fill=255)
    return np.asarray(image)>0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--images',type=int,default=6);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l)for l in(args.data/'train_verified.jsonl').open()]
    selected=[];seen=set()
    for row in rows:
        f=row['target_message']['tool_calls'][0]['function']
        if f['name']!='python'or row['meta']['original_scene_id']in seen:continue
        selected.append(row);seen.add(row['meta']['original_scene_id'])
        if len(selected)==args.images:break
    model,processor=load(SimpleNamespace(model=str(ROOT.parent/'robojudge_v4q/models/Qwen3.5-9B'),
        adapter=args.adapter,mode='generate',image_max_pixels=262144));model.eval()
    stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');results=[]
    for row in selected:
        inputs=encode(processor,row,model.device)
        with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=1800,do_sample=False,use_cache=True,eos_token_id=stop)
        tokens=ids[0,inputs['input_ids'].shape[1]:];text=processor.tokenizer.decode(tokens,skip_special_tokens=False)
        result={'id':row['id'],'stopped':bool(len(tokens)and int(tokens[-1])==stop),'tokens':len(tokens),
                'scope':'Training-prefix diagnostic only, not held-out editing.'}
        folder=args.out/row['meta']['trajectory_id'];folder.mkdir(exist_ok=True);(folder/'generation.txt').write_text(text)
        try:
            name,arguments=parse_call(text)
            if name!='python'or not result['stopped']:raise ValueError('Incomplete Python action.')
            actual=polygons(arguments['code']);target=polygons(row['target_message']['tool_calls'][0]['function']['arguments']['code'])
            reference=args.data/'executions'/row['meta']['trajectory_id']
            box=json.loads((reference/'region.json').read_text())['source_bbox_pixels']
            with Image.open(reference/'source.png')as image:size=image.size
            a=mask(actual,box,size);b=mask(target,box,size)
            intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
            result.update(mask_iou=intersection/max(1,union),target_coverage=intersection/max(1,int(b.sum())),
                predicted_points=sum(map(len,actual)),reference_points=sum(map(len,target)))
            Image.fromarray((a*255).astype(np.uint8)).save(folder/'predicted_mask.png')
            Image.fromarray((b*255).astype(np.uint8)).save(folder/'reference_mask.png')
        except Exception as error:result['parse_or_geometry_error']=type(error).__name__
        results.append(result);print(json.dumps(result),flush=True);(args.out/'results.json').write_text(json.dumps(results,indent=2))
    valid=[r for r in results if'mask_iou'in r]
    summary={'samples':len(results),'parsed_masks':len(valid),'mean_training_mask_iou':sum(r['mask_iou']for r in valid)/len(valid)if valid else None,
        'note':'The taught polygon representation is measured; other valid editing methods are not scored here. Training prefixes contain gold preceding crop actions, not gold next-action tokens.'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print('TRAINING_MASK_DIAGNOSTIC',json.dumps(summary),flush=True)


if __name__=='__main__':main()
