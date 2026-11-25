"""Compare a teacher prefix with the same actual autonomous source/crop history."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode
from pixel.agent_protocol import assistant_call,parse_call


def compare(processor,a,b):
    results={}
    for key in a:
        if not isinstance(a[key],torch.Tensor):continue
        same=a[key].shape==b[key].shape and torch.equal(a[key],b[key])
        value={'same':same,'teacher_shape':list(a[key].shape),'actual_shape':list(b[key].shape)}
        if key=='input_ids'and not same:
            x=a[key][0].tolist();y=b[key][0].tolist();index=next((i for i,(u,v)in enumerate(zip(x,y))if u!=v),min(len(x),len(y)))
            value.update(first_different_token=index,
                teacher_text=processor.tokenizer.decode(x[max(0,index-30):index+40]),
                actual_text=processor.tokenizer.decode(y[max(0,index-30):index+40]))
        results[key]=value
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--rollout',type=Path,required=True);parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    rows=[json.loads(l)for l in args.data.open()]
    teacher=next(r for r in rows if r['target_message']['tool_calls'][0]['function']['name']=='python')
    trajectory=json.loads((args.rollout/'trajectory.json').read_text())
    actual=copy.deepcopy(teacher);actual['messages']=trajectory['messages'][:len(teacher['messages'])]
    canonical=copy.deepcopy(actual)
    for index,message in enumerate(canonical['messages']):
        if message['role']=='assistant'and not message.get('tool_calls'):
            name,arguments=parse_call(message['content']);canonical['messages'][index]=assistant_call(name,arguments)
    processor=AutoProcessor.from_pretrained(args.adapter)
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    a=encode(processor,teacher,'cpu');b=encode(processor,actual,'cpu');c=encode(processor,canonical,'cpu')
    raw=compare(processor,a,b);normalized=compare(processor,a,c)
    result={'raw_autonomous_history':raw,'parsed_structured_autonomous_history':normalized,
        'raw_all_tensors_equal':all(v['same']for v in raw.values()),
        'canonical_all_tensors_equal':all(v['same']for v in normalized.values()),
        'actual_trajectory_sha256':hashlib.sha256((args.rollout/'trajectory.json').read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
