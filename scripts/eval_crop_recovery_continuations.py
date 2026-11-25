"""Evaluate model-generated continuation after correct/incomplete observed crops.

The four photographs are existing training examples. Final-reference code is
executed only in the separate evaluator workspace; it is never an actor input.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'train'));sys.path.insert(0,str(ROOT/'scripts'))
from run_pixel_sft import load,encode
from run_agent_demo import rollout
from audit_generated_prefix_pixels import initialize_history,run_action,function_of,pixel_comparison
from pixel.decoding import configure_decoding,DecodePolicyModel
from pixel.image_observations import load_observation


def fingerprints(inputs):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype),
        'sha256':hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for k,v in inputs.items()if isinstance(v,torch.Tensor)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--decode-policy',choices=['deterministic','fp32_state'],default='deterministic')
    p.add_argument('--prefixes',type=Path,default=ROOT/'outputs/agent_sft_en_27b_crop_recovery_probe_20261125/prefixes.jsonl')
    args=p.parse_args()
    if args.out.exists():raise ValueError('Choose a fresh output directory.')
    args.out.mkdir(parents=True)
    rows=[json.loads(l)for l in args.prefixes.read_text().splitlines()if l.strip()]
    correct={r['id'].removesuffix('_correct_roi'):r for r in rows if r['id'].endswith('_correct_roi')}
    configure_decoding(args.decode_policy)
    model_path=ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B'
    model,processor=load(SimpleNamespace(model=str(model_path),adapter=args.adapter,mode='generate',image_max_pixels=262144))
    model=DecodePolicyModel(model,args.decode_policy)
    cases=[]
    for row in rows:
        base_id=row['id'].removesuffix('_correct_roi').removesuffix('_partial_roi')
        reference=correct[base_id]
        base=args.out/row['id'];base.mkdir()
        source_block=next(b for m in row['messages']if m['role']=='user'for b in m['content']if b['type']=='image')
        source_path=base/'source_input.png';load_observation(source_block).save(source_path)
        expected=fingerprints(encode(processor,{'messages':row['messages'],'tools':row['tools']},model.device))
        instruction=next(m['edit_instruction']for m in row['messages']if m['role']=='user')
        folder=args.out/row['id']/'actor'
        summary,_=rollout(model,processor,source_path,instruction,folder,max_steps=8,max_new_tokens=4096,
            temperature=0.,model_name=str(model_path),adapter=args.adapter,system_profile='clean_edit',resume_prefix=row)
        trajectory=json.loads((folder/'trajectory.json').read_text())
        actual=trajectory['turns'][0]['input_fingerprints']
        if expected!=actual:raise ValueError('Actor prefix tensors differ from verified training prefix.')
        # Create the evaluator's target only after the actor is done. It is never
        # present in the actor workspace, prompt or model-generation prefix.
        source,step=initialize_history(reference,base/'reference')
        name,arguments=function_of(reference['target_message'])
        result=run_action(name,arguments,base/'reference',step)
        if not result.get('valid_native_rgb'):raise ValueError('Reference execution failed.')
        reference_rgb=np.load(base/'reference/edited_rgb.npy',allow_pickle=False)
        case={'id':row['id'],'condition':'partial'if row['id'].endswith('_partial_roi')else'correct',
            'complete':summary['complete'],'actions':summary['actions'],'native_prefix_equal':True,
            'history':summary['supplied_prefix_diagnostic'],'not_heldout_or_initial_task_autonomous':True}
        if summary['complete']:
            prediction=np.load(folder/'edited_rgb.npy',allow_pickle=False)
            case['pixels']=pixel_comparison(source,reference_rgb,prediction)
        cases.append(case)
        (args.out/'results.json').write_text(json.dumps(cases,indent=2))
        print('RECOVERY_CASE',json.dumps(case),flush=True)
    report={'cases':cases,'completed':sum(r['complete']for r in cases),
        'exact_reference_rgb':sum(r.get('pixels',{}).get('exact_reference_rgb',False)for r in cases),
        'adapter':args.adapter,'decode_policy':args.decode_policy,'scope':'eight model-generated continuations on four existing training photographs'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    print('RECOVERY_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='cases'}),flush=True)


if __name__=='__main__':main()
