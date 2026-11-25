"""Check actual native tensors before/after fresh history restoration, without weights."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode
from pixel.image_observations import load_observation
from pixel.prefix_replay import replay_prefix


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Use a fresh audit output.')
    args.out.mkdir(parents=True)
    processor=AutoProcessor.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    rows=[json.loads(l)for l in (ROOT/'outputs/agent_sft_en_27b_crop_recovery_probe_20261125/prefixes.jsonl').read_text().splitlines()]
    cases=[]
    for row in rows:
        folder=args.out/row['id'];folder.mkdir()
        block=next(b for m in row['messages']if m['role']=='user'for b in m['content']if b['type']=='image')
        load_observation(block).save(folder/'source.png')
        proof=replay_prefix(row,folder);messages=proof.pop('rebound_messages')
        before=encode(processor,row,'cpu')
        after=encode(processor,{'messages':messages,'tools':row['tools']},'cpu')
        if before.keys()!=after.keys()or not all(torch.equal(before[k],after[k])for k in before):
            raise ValueError('Restored native prefix differs: '+row['id'])
        cases.append({'id':row['id'],'native_tensors_equal':True,'tensor_fields':list(before),'proof':proof})
    (args.out/'summary.json').write_text(json.dumps({'cases':cases,'passed':len(cases)},indent=2))
    print('ALL_NATIVE_PREFIX_RESTORATIONS_EQUAL',len(cases),flush=True)


if __name__=='__main__':main()
