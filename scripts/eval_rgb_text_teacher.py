"""One-shot GPT-6.1 editing across controlled input modalities; no target or tools."""
import argparse
import ast
import concurrent.futures as cf
import hashlib
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pixel.rgb_text import decode_rgb
from pixel.rgb_text_prompts import EDIT_SYSTEM,make_instruction
from pixel.rgb_text_eval import pixel_metrics
from build_teacher_magicbrush_inpaint import parse_json,run_code
from build_magicbrush_target_informed import looks_hardcoded
from pixel.teacher_api import make_client,judge_edit
from pixel.persistent_io import atomic_text


def call_teacher(body,key):
    final=None;deltas=[];event=''
    with requests.Session()as session:
        session.trust_env=False
        with session.post('http://127.0.0.1:9589/v1/responses',json=body,
                          headers={'Authorization':'Bearer '+key},stream=True,timeout=(10,700))as response:
            response.raise_for_status();response.encoding='utf-8'
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('event:'):event=line[6:].strip()
                if not line.startswith('data:'):continue
                try:value=json.loads(line[5:].strip())
                except json.JSONDecodeError:continue
                kind=value.get('type',event)
                if kind=='response.output_text.delta':deltas.append(value.get('delta',''))
                elif kind=='response.completed':final=value.get('response')
                elif kind in('response.failed','error'):
                    raise RuntimeError('Teacher API failure: '+str(value.get('error')or value.get('response',{}).get('error'))[:200])
    if not final or final.get('status')!='completed' or final.get('model')!=body['model']:
        raise RuntimeError('No completed response from requested teacher')
    text=''.join(part.get('text','')for item in final.get('output',[])for part in item.get('content',[])if part.get('type')=='output_text')
    text=text or ''.join(deltas)
    if not text:raise RuntimeError('Teacher final response is empty')
    return text,final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'outputs/rgb_text_gpt61_smoke_v1'))
    parser.add_argument('--also-instruction-only',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=['rgb_text','image_only','image_rgb','instruction_only'])
    parser.add_argument('--cache-key',action='store_true')
    parser.add_argument('--tasks',default=str(ROOT/'data/rgb_text_smoke_v1/tasks.jsonl'))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--expected-tasks',type=int,default=None)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(l)for l in Path(args.tasks).read_text().splitlines()]
    modes=args.modes or(['rgb_text','instruction_only']if args.also_instruction_only else['rgb_text'])
    modality_state={'modes':modes,'image_inputs_by_mode':{m:int(m in('image_only','image_rgb'))for m in modes},'target_in_model_input':False}
    atomic_text(out/'run_state.json',json.dumps({'state':'running','pid':os.getpid(),'model':'traex/GPT-6.1-Sol',**modality_state}))
    key=(ROOT.parent/'traex_api_proxy/proxy_logs_rgb_gpt61/local.key').read_text().strip()
    client=make_client();records=[]
    if(out/'results.jsonl').is_file():
        records=[json.loads(l)for l in(out/'results.jsonl').read_text().splitlines()if l.strip()]
    def execute(task,mode):
        folder=out/mode/task['id'];folder.mkdir(parents=True,exist_ok=True)
        record={'id':task['id'],'model':'traex/GPT-6.1-Sol','mode':mode,'dataset':task['dataset'],
                'instruction':task['instruction'],'resolution':[task['width'],task['height']],
                'image_inputs':int(mode in('image_only','image_rgb')),'target_in_model_input':False,'tools_in_model_input':False,
                'evaluation_only':True,'parsed':False,'source_image':task['image'],'target_image':task['target_image']}
        start=time.monotonic()
        try:
            decoded=np.asarray(decode_rgb(Path(task['rgb_text']).read_text()))
            with Image.open(task['image'])as image:assert np.array_equal(decoded,np.asarray(image.convert('RGB')))
            record['source_rgb_sha256']=hashlib.sha256(decoded.tobytes()).hexdigest()
            instruction=make_instruction(task,'vl'if mode=='image_only'else'rgb_text'if mode=='image_rgb'else mode)
            if mode=='image_rgb':
                instruction=instruction.replace('No image is attached. The complete source is supplied below as lossless numeric RGB text.',
                    'The attached source image and the complete lossless numeric RGB text below represent the identical pixels.')
            content=[{'type':'input_text','text':instruction}]
            if record['image_inputs']:
                with Image.open(task['image'])as image:
                    buffer=io.BytesIO();image.convert('RGB').save(buffer,format='PNG')
                content.insert(0,{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()})
            body={'model':record['model'],'input':[
                {'role':'system','content':[{'type':'input_text','text':EDIT_SYSTEM}]},
                {'role':'user','content':content}],
                'reasoning':{'effort':'high'},'max_output_tokens':8192,'stream':True}
            if args.cache_key:body['prompt_cache_key']=f'gpt61_modalities_v1:{mode}:{task["id"]}'
            if not record['image_inputs']:assert all(part['type']=='input_text'for message in body['input']for part in message['content'])
            atomic_text(folder/'request.json',json.dumps(body,ensure_ascii=False,indent=2))
            record['request_characters']=len(instruction)+len(EDIT_SYSTEM)
            record['generation_api_calls']=1
            print('START',task['id'],mode,record['request_characters'],flush=True)
            raw,response=call_teacher(body,key)
            atomic_text(folder/'completion.txt',raw)
            record['teacher_usage']=response.get('usage')
            atomic_text(folder/'response.json',json.dumps(response,ensure_ascii=False,indent=2))
            parsed=parse_json(raw);record['parsed']=True;code=parsed.get('executable_code','')
            blocked,reason=looks_hardcoded(code)
            allowed={'PIL','numpy','cv2','math','colorsys','random'}
            for node in ast.walk(ast.parse(code)):
                modules=([a.name for a in node.names]if isinstance(node,ast.Import)else
                         [node.module or'']if isinstance(node,ast.ImportFrom)else[])
                if any(m.split('.')[0]not in allowed for m in modules):blocked,reason=True,'unsupported_library'
            result={'status':'rejected','error':reason,'output':None}if blocked else run_code(code,Path(task['image']),folder/'edited.png')
            record['execution']=result
            if result['status']=='passed':
                try:record['metrics']=pixel_metrics(task['image'],task['target_image'],result['output'])
                except Exception as exc:record['metrics_error']=type(exc).__name__+': '+str(exc)[:200]
                try:record['judge']=judge_edit(client,task['image'],result['output'],task['instruction'],strict_extent=True)
                except Exception as exc:record['judge_pending']=True;record['judge_error']=type(exc).__name__
        except Exception as exc:record['error']=type(exc).__name__+': '+str(exc)[:250]
        record['elapsed_seconds']=round(time.monotonic()-start,2)
        atomic_text(folder/'record.json',json.dumps(record,ensure_ascii=False,indent=2))
        return record
    summary_modes=list(dict.fromkeys([r['mode']for r in records]+modes))
    existing={(r['id'],r['mode'])for r in records}
    # Separate conditions use independent single responses; no repair/retry of generated code.
    with cf.ThreadPoolExecutor(max_workers=args.workers)as pool:
        # Interleave modalities per task so paired results arrive together.
        futures=[pool.submit(execute,t,m)for t in tasks for m in modes if(t['id'],m)not in existing]
        for future in cf.as_completed(futures):
            record=future.result();records.append(record)
            atomic_text(out/'results.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in records))
            summary={}
            for mode in summary_modes:
                subset=[r for r in records if r['mode']==mode]
                summary[mode]={'completed':len(subset),'expected':args.expected_tasks or len(tasks),'executed':sum(r.get('execution',{}).get('status')=='passed'for r in subset),
                               'judge_pass':sum(r.get('judge',{}).get('pass')is True for r in subset),
                               'judge_pending':sum(bool(r.get('judge_pending'))for r in subset),'errors':sum('error'in r for r in subset)}
            atomic_text(out/'summary.json',json.dumps(summary,indent=2))
            print('RESULT',record['id'],record['mode'],record.get('execution',{}).get('status'),record.get('judge',{}).get('pass'),record.get('error'),flush=True)
    atomic_text(out/'run_state.json',json.dumps({'state':'completed','model':'traex/GPT-6.1-Sol',**modality_state,'results':len(records)}))


if __name__=='__main__':main()
