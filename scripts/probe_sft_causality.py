"""Check whether future target tokens affect logits before the assistant action."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import load,encode


def main():
    parser=argparse.ArgumentParser()
    for flag in ['model','adapter','data','out']:parser.add_argument('--'+flag,required=True)
    args=parser.parse_args()
    model,processor=load(SimpleNamespace(model=args.model,adapter=args.adapter,mode='generate',image_max_pixels=262144))
    model.eval()
    rows=[json.loads(l)for l in Path(args.data).read_text().splitlines()if l.strip()]
    row=next(r for r in rows if r['target_message']['tool_calls'][0]['function']['name']=='python')
    prompt=encode(processor,row,model.device)
    full=encode(processor,row,model.device,True)
    full={k:v for k,v in full.items()if k!='labels'}
    n=prompt['input_ids'].shape[1]
    altered={**full,'input_ids':full['input_ids'].clone()}
    replacement=processor.tokenizer.encode('x',add_special_tokens=False)[0]
    special=torch.tensor(processor.tokenizer.all_special_ids,device=model.device)
    suffix=altered['input_ids'][:,n:]
    suffix[~torch.isin(suffix,special)]=replacement
    def prefix_logits(inputs,index):
        with torch.no_grad():result=model(**inputs,use_cache=False)
        value=result.logits[0,index].float().detach().cpu()
        del result
        return value
    original=prefix_logits(full,n-1)
    modified=prefix_logits(altered,n-1)
    repeat=prefix_logits(full,n-1)
    shorter=prefix_logits(prompt,n-1)
    record={'id':row['id'],'prompt_tokens':n,'full_tokens':full['input_ids'].shape[1],
        'same_length_future_change_max':float((original-modified).abs().max()),
        'same_length_future_change_mean':float((original-modified).abs().mean()),
        'repeat_max':float((original-repeat).abs().max()),
        'different_length_max':float((original-shorter).abs().max()),
        'different_length_mean':float((original-shorter).abs().mean()),
        'prefix_argmax_unchanged':bool(original.argmax()==modified.argmax())}
    Path(args.out).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
