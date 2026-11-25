"""Audit actual processor tensors and next-action loss masks for every pilot row."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'train'))
from trajectory_io import encode_trajectory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--processor',required=True)
    parser.add_argument('--data',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    processor=AutoProcessor.from_pretrained(args.processor,local_files_only=True)
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    reports=[]
    for name in args.data:
        for line in Path(name).read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            prompt=encode_trajectory(processor,row,'cpu')
            full=encode_trajectory(processor,row,'cpu',True,8192)
            if full is None:raise ValueError('Pilot exceeds context budget: '+row['id'])
            n=prompt['input_ids'].shape[1]
            labels=full['labels']
            assert torch.all(labels[:,:n]==-100),row['id']
            assert torch.equal(labels[:,n:],full['input_ids'][:,n:]),row['id']
            assert labels.shape[1]>n,row['id']
            matches={}
            for key,value in prompt.items():
                if not isinstance(value,torch.Tensor):continue
                compare=full[key][:,:n]if value.shape==prompt['input_ids'].shape else full[key]
                matches[key]=value.shape==compare.shape and torch.equal(value,compare)
            assert all(matches.values()),(row['id'],matches)
            target=processor.tokenizer.decode(labels[:,n:][0].tolist(),skip_special_tokens=False)
            assert '<tool_call>'in target and '</tool_call>'in target,row['id']
            reports.append({'id':row['id'],'dataset':name,'prompt_tokens':n,
                'assistant_tokens':labels.shape[1]-n,'only_next_assistant_supervised':True,
                'prompt_full_tensors_match':matches})
    assert reports,'No actual pilot examples were audited.'
    result={'rows':len(reports),'processor':args.processor,'all_passed':True,'examples':reports}
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({'rows':len(reports),'all_passed':True}),flush=True)


if __name__=='__main__':main()
