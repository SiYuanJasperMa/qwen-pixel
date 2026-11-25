"""Check actual masks, geometry loss spans, and causal SFT inputs."""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/sft_clean_mask_v2/train.jsonl')
    parser.add_argument('--processor',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    processor=AutoProcessor.from_pretrained(args.processor,local_files_only=True)
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    rows=[json.loads(l)for l in args.data.open()]
    checks=[]
    selected=rows[:args.limit]if args.limit else rows
    if not selected:raise ValueError('No actual training rows were selected.')
    for row in selected:
        action=row['target_message']['tool_calls'][0]['function']['name']
        if action=='python'and 'polygons_roi = ['in json.dumps(row['messages']):
            if not(row.get('meta',{}).get('refinement')and row['meta'].get('supervised_actions_exclude_mistake')):
                raise ValueError('Ground-truth polygons entered the context before first prediction.')
            # A real prior wrong program is valid history; the future correct
            # repair must not already have been given to the student.
            future=row['target_message']['tool_calls'][0]['function']['arguments']['code']
            for message in row['messages']:
                for call in message.get('tool_calls',[]):
                    prior=call['function'].get('arguments',{}).get('code','')
                    if prior.strip()==future.strip():raise ValueError('Future repair code leaked into its input prefix.')
                if isinstance(message['content'],str)and future in message['content']:
                    raise ValueError('Future repair code leaked into its input prefix.')
        batch=encode(processor,row,'cpu',True,8192,1.,3.)
        if batch is None:raise ValueError('Clean row exceeds training context.')
        labels=batch['labels'];weights=batch['token_weights']
        if not(weights[labels==-100]==0).all():raise ValueError('Prompt tokens received loss weights.')
        geometric=weights[0]>1
        count=int(geometric.sum())
        function=row['target_message']['tool_calls'][0]['function']
        has_geometry=action=='python'or(action=='view_image'and'bbox_2d'in function['arguments'])
        if has_geometry and count==0:
            raise ValueError('Visual geometry has no training weight.')
        check={'id':row['id'],'action':action,'tokens':int(labels.numel()),
            'assistant_tokens':int((labels!=-100).sum()),'geometry_tokens':count,
            'geometry_text':processor.tokenizer.decode(labels[0,geometric].tolist(),skip_special_tokens=False)}
        checks.append(check);print(json.dumps(check),flush=True)
    destination=args.out or args.data.parent/'encoding_audit.json'
    destination.write_text(json.dumps(checks,indent=2))
    print('CLEAN_SFT_ENCODING_AUDIT_PASSED',flush=True)


if __name__=='__main__':main()
