"""Same real edit tasks: RGB-text pure LLM, instruction-only, or VL control."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'train'))
from pixel.rgb_text_io import load_text_decoder,make_prompt,make_instruction,EDIT_SYSTEM,encode_text_oneshot
from pixel.rgb_text import decode_rgb
from pixel.rgb_text_eval import pixel_metrics,summarize
from pixel.oneshot_io import encode_oneshot
from pixel.teacher_api import judge_edit,make_client
from build_teacher_magicbrush_inpaint import parse_json,run_code
from build_magicbrush_target_informed import looks_hardcoded
from run_pixel_sft import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--mode',choices=['rgb_text','vl'],required=True)
    p.add_argument('--also-instruction-only',action='store_true')
    p.add_argument('--max-input-tokens',type=int,default=140000)
    p.add_argument('--max-new-tokens',type=int,default=4096)
    p.add_argument('--limit',type=int,default=3)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(l)for l in Path(args.tasks).read_text().splitlines()if l.strip()][:args.limit]
    if args.mode=='rgb_text':
        model,tokenizer,audit=load_text_decoder(args.model);processor=None
    else:
        model,processor=load(SimpleNamespace(model=args.model,adapter=None,mode='generate',image_max_pixels=262144))
        tokenizer=processor.tokenizer;audit={'class':model.__class__.__name__,'mode':'vl'}
    (out/'model_audit.json').write_text(json.dumps(audit,indent=2))
    client=make_client();records=[]
    for task in tasks:
        for mode in ([args.mode,'instruction_only']if args.mode=='rgb_text'and args.also_instruction_only else[args.mode]):
            folder=out/mode/task['id'];folder.mkdir(parents=True,exist_ok=True)
            r={'id':task['id'],'mode':mode,'dataset':task['dataset'],'instruction':task['instruction'],
               'source_side':task['width'],'vision_encoder_used':mode=='vl','parsed':False}
            start=time.monotonic()
            try:
                rgb=decode_rgb(Path(task['rgb_text']).read_text())
                with Image.open(task['image'])as source:
                    assert np.array_equal(np.asarray(source.convert('RGB')),np.asarray(rgb))
                r['source_rgb_sha256']=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                if mode=='vl':
                    inputs=encode_oneshot(processor,{'id':task['id'],'system':EDIT_SYSTEM,
                        'instruction':make_instruction(task,'vl'),'image':task['image']},model.device)
                else:
                    text=make_prompt(tokenizer,task,mode)
                    (folder/'prompt.txt').write_text(text)
                    inputs=encode_text_oneshot(tokenizer,task,model.device,mode)
                    assert set(inputs)=={'input_ids','attention_mask'}
                    r['model_input_fields']=list(inputs)
                n=inputs['input_ids'].shape[1];r['input_tokens']=n
                if n>args.max_input_tokens:raise ValueError(f'Input token budget exceeded: {n}')
                print('GENERATING',task['id'],mode,n,flush=True)
                with torch.inference_mode():
                    ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,
                        use_cache=True,logits_to_keep=1,eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
                        pad_token_id=tokenizer.pad_token_id)[0,n:]
                raw=tokenizer.decode(ids,skip_special_tokens=True)
                r['output_tokens']=len(ids);(folder/'completion.txt').write_text(raw)
                parsed=parse_json(raw);code=parsed.get('executable_code','');r['parsed']=True
                rejected,reason=looks_hardcoded(code)
                execution=({'status':'rejected','error':reason,'output':None}if rejected
                           else run_code(code,Path(task['image']),folder/'edited.png'))
                r['execution']=execution
                if execution['status']=='passed':
                    r['metrics']=pixel_metrics(task['image'],task['target_image'],execution['output'])
                    try:r['judge']=judge_edit(client,task['image'],execution['output'],task['instruction'],strict_extent=True)
                    except Exception as exc:r['judge_pending']=True;r['judge_error']=type(exc).__name__+': '+str(exc)[:200]
            except torch.OutOfMemoryError as exc:
                r['error']='cuda_oom';torch.cuda.empty_cache()
            except Exception as exc:r['error']=type(exc).__name__+': '+str(exc)[:300]
            r['elapsed_seconds']=round(time.monotonic()-start,2)
            records.append(r)
            with(out/'results.jsonl').open('a')as f:f.write(json.dumps(r)+'\n')
            (out/'summary.json').write_text(json.dumps(summarize(records,audit),indent=2))
            print('RESULT',json.dumps(r),flush=True)
            if r.get('error')=='cuda_oom':
                # Do not silently alter the resolution or provide a hidden image.
                print('STOP_AFTER_OOM_RESOLUTION_UNCHANGED',flush=True);return


if __name__=='__main__':main()
