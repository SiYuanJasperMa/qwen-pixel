"""Check whether the instruct student can locate editing targets before coding.

This is a diagnostic within the edit workflow, not an added training task.
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode,load


def main():
    out=ROOT/'outputs/sft_audit/localization'
    out.mkdir(parents=True,exist_ok=True)
    model_name=ROOT.parent/'robojudge_v4q/models/Qwen3.5-9B'
    model,processor=load(SimpleNamespace(model=str(model_name),adapter=None,
        mode='generate',image_max_pixels=262144))
    model.eval()
    tasks=[json.loads(line)for line in(ROOT/'data/sft_seven_tasks_v1/tasks.jsonl').open()]
    tasks=[t for t in tasks if t['split']=='val']
    results=[]
    for task in tasks:
        question=('Locate the object that must be edited for this instruction: '+task['instruction']+
            '\nReturn only JSON with bbox_2d: [x1,y1,x2,y2], using normalized 0-1000 image coordinates. '
            'The box should enclose the visible target object. Do not edit the image or generate Python yet.')
        row={'id':task['id'],'messages':[{'role':'system','content':'You identify the target region for image editing.'},
            {'role':'user','content':[{'type':'image','image':task['image']},{'type':'text','text':question}]}],
            'tools':[]}
        inputs=encode(processor,row,model.device)
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=150,do_sample=False,
                eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        text=processor.tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        result={'id':task['id'],'instruction':task['instruction'],'text':text}
        match=re.search(r'"bbox_2d"\s*:\s*(\[[^\]]+\])',text)
        folder=out/task['id'];folder.mkdir(exist_ok=True)
        source=Image.open(task['image']).convert('RGB');source.save(folder/'source.png')
        if match:
            normalized=json.loads(match[1]);w,h=source.size
            if len(normalized)==4:
                box=[max(0,min(w if i%2==0 else h,round(float(value)*(w if i%2==0 else h)/1000)))
                     for i,value in enumerate(normalized)]
                if box[2]>box[0]and box[3]>box[1]:
                    crop=source.crop(tuple(box));crop.save(folder/'crop.png')
                    overlay=source.copy();ImageDraw.Draw(overlay).rectangle(tuple(box),outline='red',width=3)
                    overlay.save(folder/'bbox.png')
                    result.update(bbox_2d=normalized,bbox_pixels=box)
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
