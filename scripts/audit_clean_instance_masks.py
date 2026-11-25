"""Review full-object mask semantics without producing new training images."""
import ast
import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.teacher_api import make_client,image_block,MODEL


def review(task,out,data):
    identity=task['original_scene_id'];path=out/(identity+'.json')
    if path.exists():return json.loads(path.read_text())
    folder=data/'executions'/task['id']
    with Image.open(folder/'source.png')as image:source=image.convert('RGB')
    region=json.loads((folder/'region.json').read_text())['source_bbox_pixels']
    script=(folder/'action_01.py').read_text();tree=ast.parse(script)
    polygons=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='polygons_roi'for t in n.targets))
    x0,y0,x1,y1=region;mask=Image.new('L',source.size,0);draw=ImageDraw.Draw(mask)
    for polygon in polygons:draw.polygon([(x0+u*(x1-x0),y0+v*(y1-y0))for u,v in polygon],fill=255)
    a=np.asarray(source).astype(float);m=np.asarray(mask)>0
    overlay=a.copy();overlay[m]=overlay[m]*.6+np.array([255,20,20])*.4
    case=out/identity;case.mkdir(exist_ok=True)
    source.save(case/'source.png');Image.fromarray(overlay.astype(np.uint8)).save(case/'overlay.png')
    mask.crop(tuple(region)).save(case/'mask_crop.png');source.crop(tuple(region)).save(case/'source_crop.png')
    prompt=('Verify a training mask for a real-image edit. Requested target: '+task['instruction']+
        '\nThe annotated instance source box is '+json.dumps(region)+
        '. Image1 source, image2 red mask overlay, image3 target crop, image4 binary mask crop. '
        'A box identifies the instance but is not itself a mask. Check that the mask covers essentially all visible target surfaces and excludes unrelated or occluding objects. Other same-category instances are not requested targets. '
        'Return JSON: coverage_score 1-5, exclusion_score 1-5, pass boolean, reason. Partial masks, masks covering glasses/items on a table, or the wrong instance must fail. This is mask quality, not edit strength or Qwen evaluation.')
    response=make_client().chat.completions.create(model=MODEL,
        messages=[{'role':'user','content':[{'type':'text','text':prompt},
            image_block(case/'source.png'),image_block(case/'overlay.png'),
            image_block(case/'source_crop.png'),image_block(case/'mask_crop.png')]}],
        response_format={'type':'json_object'},reasoning_effort='low',max_completion_tokens=4096)
    judgment=json.loads(response.choices[0].message.content or '{}')
    accepted=(judgment.get('pass')is True and float(judgment['coverage_score'])>=4 and float(judgment['exclusion_score'])>=4)
    result={'image_id':identity,'accepted':accepted,'judgment':judgment,'source_task':task['id'],
        'new_teacher_generation':False,'source_sha256':hashlib.sha256((folder/'source.png').read_bytes()).hexdigest()}
    path.write_text(json.dumps(result,indent=2));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/sft_clean_mask_v2')
    parser.add_argument('--concurrency',type=int,default=8);args=parser.parse_args()
    data=args.data.resolve();out=data/'mask_reviews';out.mkdir(exist_ok=True)
    tasks=[json.loads(l)for l in(data/'tasks.jsonl').open()];unique={}
    for task in tasks:unique.setdefault(task['original_scene_id'],task)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)as pool:
        pending={pool.submit(review,t,out,data):t for t in unique.values()}
        for future in concurrent.futures.as_completed(pending):
            task=pending[future]
            try:result=future.result()
            except Exception as error:result={'image_id':task['original_scene_id'],'accepted':False,'review_error':type(error).__name__}
            results.append(result);print(json.dumps(result),flush=True)
    summary={'images_reviewed':len(results),'accepted':sum(r['accepted']for r in results),
        'review_errors':sum('review_error'in r for r in results),'new_teacher_generation':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('MASK_SEMANTICS_REVIEW_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
