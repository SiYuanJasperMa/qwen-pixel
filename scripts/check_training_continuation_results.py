"""Check real 27B uninterrupted/resumed updates and edited native RGB artifacts."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.image_observations import load_observation


def metrics(folder):
    records={}
    for line in(folder/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);records[row['step']]=row
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    expected=metrics(args.root/'uninterrupted')
    actual={**metrics(args.root/'interrupted'),**metrics(args.root/'resumed')}
    if expected.keys()!=actual.keys():raise ValueError('Resumed updates missing or repeated.')
    for step in expected:
        for key in ['sample_ids','loss','supervised_tokens','lr']:
            if expected[step][key]!=actual[step][key]:raise ValueError(f'Continuation differs at{step}: {key}')
    a=load_file(args.root/'uninterrupted/checkpoint-12/adapter_model.safetensors')
    b=load_file(args.root/'resumed/checkpoint-12/adapter_model.safetensors')
    if a.keys()!=b.keys()or not all(torch.equal(a[k],b[k])for k in a):
        raise ValueError('Real resumed LoRA weights differ.')
    rgb_checks=[]
    rows=[json.loads(l)for l in(ROOT/'data/sft_exact_replay_v1/train.jsonl').read_text().splitlines()]
    block=next(b for m in reversed(rows[-1]['messages'])if isinstance(m.get('content'),list)
               for b in m['content']if b['type']=='image')
    reference=np.asarray(load_observation(block))
    for step in [8,12]:
        a=np.load(args.root/f'uninterrupted/checkpoint-{step}/agent_validation/coco_agent_008/edited_rgb.npy',allow_pickle=False)
        b=np.load(args.root/f'resumed/checkpoint-{step}/agent_validation/coco_agent_008/edited_rgb.npy',allow_pickle=False)
        if not np.array_equal(a,b):raise ValueError('Resumed native editing differs.')
        if not np.array_equal(a,reference):raise ValueError('Both policies regressed from accepted reference RGB.')
        rgb_checks.append({'step':step,'native_rgb_exact':True})
    result={'updates_exact':len(actual),'final_adapter_tensors_exact':True,'native_edits':rgb_checks,
            'scope':'real27B optimizer/sampler/RNG continuation; engineering regression, not held-out quality'}
    (args.root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
