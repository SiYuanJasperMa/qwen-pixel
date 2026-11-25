"""Test reusing image-teacher labels after converting student input to RGB text."""
import ast
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pixel.rgb_text import encode_rgb,decode_rgb
from pixel.rgb_text_prompts import EDIT_SYSTEM,make_instruction
from pixel.rgb_text_eval import pixel_metrics
from pixel.teacher_api import make_client,judge_edit
from pixel.persistent_io import atomic_text
from build_teacher_magicbrush_inpaint import parse_json,run_code


def read(path):return[json.loads(l)for l in path.read_text().splitlines()if l.strip()]


def main():
    out=ROOT/'outputs/rgb_label_reuse_audit_v1';out.mkdir(exist_ok=True)
    sft=read(ROOT/'data/editing_growth_v1/sft/train.jsonl')+read(ROOT/'data/editing_growth_v1/gpt62/sft/train.jsonl')
    originals={r['id']:r for folder in ['data/editing_growth_v1','data/editing_growth_v1/gpt62']for r in read(ROOT/folder/'judged.jsonl')}
    buckets=defaultdict(list)
    for row in sft:
        if row['id']in originals:buckets[row['meta']['dataset']].append(row)
    selected=[]
    for ds,rows in sorted(buckets.items()):
        random.Random(617).shuffle(rows)
        # Two direct traces per dataset, plus a TI trace when available.
        rft=[r for r in rows if r['meta']['generation_mode']=='rft']
        ti=[r for r in rows if r['meta']['generation_mode']=='ti']
        selected+=rft[:2]+(ti[:1]if ti else rft[2:3])
    tok=AutoTokenizer.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    client=make_client()
    def audit(row):
        original=originals[row['id']];answer=json.loads(row['answer']);code=answer['executable_code']
        folder=out/row['id'];folder.mkdir(exist_ok=True)
        source=folder/'source128.png';target=folder/'target128.png'
        for src,dst in [(row['image'],source),(original['target_image'],target)]:
            with Image.open(src)as image:image.convert('RGB').resize((128,128),Image.Resampling.LANCZOS).save(dst)
        with Image.open(source)as image:
            text=encode_rgb(image);assert np.array_equal(np.asarray(image.convert('RGB')),np.asarray(decode_rgb(text)))
        rgb=folder/'source.rgb.txt';atomic_text(rgb,text)
        task={'id':row['id'],'instruction':row['meta']['instruction'],'rgb_text':str(rgb),'width':128,'height':128}
        user=make_instruction(task,'rgb_text')
        prompt=tok.apply_chat_template([{'role':'system','content':EDIT_SYSTEM},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok.encode(prompt,add_special_tokens=False))
        label_tokens=len(tok.encode(row['answer']+'<|im_end|>',add_special_tokens=False))
        record={'id':row['id'],'dataset':row['meta']['dataset'],'teacher':row['meta']['teacher_backend'],'mode':row['meta']['generation_mode'],
                'input_tokens':n,'answer_tokens':label_tokens,'fits_140k':n+label_tokens<=140000,
                'source_rgb_lossless':True,'native_accepted':True,'literal_512':bool(re.search(r'\b512\b',code)),
                'target_reference_in_label':bool(re.search(r'target image|reference image|second image|image 2',json.dumps({k:v for k,v in answer.items()if k!='executable_code'}),re.I))}
        result=run_code(code,source,folder/'edited128.png');record['execution']=result
        if result['status']=='passed':
            try:
                record['pixel_metrics']=pixel_metrics(source,target,result['output'])
                retained=original.get('retained_output')
                if retained and Path(retained).is_file():
                    with Image.open(retained)as im:a=np.asarray(im.convert('RGB').resize((128,128),Image.Resampling.LANCZOS),dtype=np.float32)
                    with Image.open(result['output'])as im:b=np.asarray(im.convert('RGB'),dtype=np.float32)
                    record['scale_replay_l1']=float(np.abs(a-b).mean()/255.)
                record['judge128']=judge_edit(client,str(source),result['output'],task['instruction'],strict_extent=True)
            except Exception as exc:record['eval_error']=type(exc).__name__+': '+str(exc)[:180]
        record['reuse_eligible']=record.get('judge128',{}).get('pass')is True and record['fits_140k']and not record['target_reference_in_label']
        if record['reuse_eligible']:
            example={'id':row['id']+'__rgb128','task':row['task'],'system':EDIT_SYSTEM,'instruction':task['instruction'],
                     'execution_image':str(source),'rgb_text':str(rgb),'width':128,'height':128,'answer':row['answer'],
                     'meta':{**row['meta'],'student_input_mode':'rgb_text','execution_source_resolution':128,
                             'code_label_reused_unchanged':True,'resolution_revalidated':True,'parent_sft_id':row['id'],
                             'original_teacher_input_modality':'image','rgb_text_is_model_input':True}}
            atomic_text(folder/'rgb_sft_example.json',json.dumps(example,ensure_ascii=False,indent=2))
        atomic_text(folder/'audit.json',json.dumps(record,ensure_ascii=False,indent=2))
        print('AUDIT',record['dataset'],row['id'],result['status'],record.get('judge128',{}).get('pass'),record.get('scale_replay_l1'),flush=True)
        return record
    results=[]
    with cf.ThreadPoolExecutor(max_workers=3)as pool:
        for result in pool.map(audit,selected):results.append(result)
    report={'sample_count':len(results),'datasets':dict(Counter(r['dataset']for r in results)),
            'native_accepted':len(results),'executed128':sum(r['execution']['status']=='passed'for r in results),
            'semantic_pass128':sum(r.get('judge128',{}).get('pass')is True for r in results),
            'reuse_eligible':sum(r['reuse_eligible']for r in results),'judge_or_shape_errors':sum('eval_error'in r for r in results),
            'results':results}
    atomic_text(out/'report.json',json.dumps(report,ensure_ascii=False,indent=2))
    print('SUMMARY',json.dumps({k:v for k,v in report.items()if k!='results'}),flush=True)


if __name__=='__main__':main()
