"""Compare actual training tensors against checkpoint-processor reload tensors."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from trajectory_io import encode_trajectory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    original=AutoProcessor.from_pretrained(args.model)
    restored=AutoProcessor.from_pretrained(args.checkpoint)
    for processor in (original,restored):
        processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    report={'original_image_processor':type(original.image_processor).__name__,
        'restored_image_processor':type(restored.image_processor).__name__,
        'original_processor':type(original).__name__,'restored_processor':type(restored).__name__,
        'examples':[]}
    rows=[json.loads(l)for l in Path(args.data).open()][:4]
    for row in rows:
        first=encode_trajectory(original,row,'cpu',True,8192)
        second=encode_trajectory(restored,row,'cpu',True,8192)
        if first is None or second is None:raise ValueError('Audit example exceeds its context.')
        keys=set(first)|set(second);comparison={}
        for key in sorted(keys):
            a,b=first.get(key),second.get(key)
            equal=a is not None and b is not None and a.shape==b.shape and torch.equal(a,b)
            info={'equal':equal,'original_shape':list(a.shape)if a is not None else None,
                  'restored_shape':list(b.shape)if b is not None else None}
            if a is not None and b is not None and a.shape==b.shape:
                info['max_absolute_difference']=float((a.float()-b.float()).abs().max())
            comparison[key]=info
        report['examples'].append({'id':row['id'],'tensors':comparison})
        print(json.dumps(report['examples'][-1]),flush=True)
    report['all_tensors_identical']=all(info['equal']for item in report['examples']for info in item['tensors'].values())
    Path(args.out).write_text(json.dumps(report,indent=2))
    print('PROCESSOR_PARITY',report['all_tensors_identical'],flush=True)


if __name__=='__main__':main()
