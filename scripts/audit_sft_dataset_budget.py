"""Check every supervised action fits both SFT context and actor output budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode
from pixel.sft_contract import training_contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--context',type=int,default=8192);parser.add_argument('--output-ceiling',type=int,default=4096)
    parser.add_argument('--model',type=Path,default=ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    args=parser.parse_args();rows=[json.loads(l)for l in args.data.open()]
    processor=AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    contract=training_contract(rows,262144)
    results=[]
    for row in rows:
        batch=encode(processor,row,'cpu',True,args.context,1.,1.)
        if batch is None:raise ValueError('SFT row would be silently skipped for context overflow: '+row['id'])
        total=int(batch['input_ids'].shape[1]);supervised=int((batch['labels']!=-100).sum())
        if supervised>args.output_ceiling:raise ValueError('Teacher action cannot fit actor output ceiling: '+row['id'])
        if not supervised:raise ValueError('Missing assistant labels: '+row['id'])
        results.append({'id':row['id'],'context_tokens':total,'assistant_tokens':supervised})
    summary={'records':len(results),'context_limit':args.context,'actor_output_ceiling':args.output_ceiling,
        'max_context_tokens':max(r['context_tokens']for r in results),
        'max_assistant_tokens':max(r['assistant_tokens']for r in results),'context_overflows':0,'unreachable_action_targets':0,
        'dataset_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'model':str(args.model),
        'runtime_contract_verified':bool(contract)}
    out=args.data.parent/'dataset_budget_audit.json';out.write_text(json.dumps({'summary':summary,'rows':results},indent=2))
    print('ALL_SFT_ROWS_BUDGET_AUDIT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
