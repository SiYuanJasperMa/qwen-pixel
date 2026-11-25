"""Match supervision exactly, changing only observed semantic-error history."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import execute_action,initial_messages,validate_outputs
from pixel.image_observations import load_observation


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=''.join(json.dumps(r)+'\n'for r in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Choose fresh comparison data.')
    args.out.mkdir(parents=True)
    normal=[json.loads(l)for l in(ROOT/'data/agent_sft_en_27b_37photos_20261125/train.jsonl').read_text().splitlines()]
    repairs=[json.loads(l)for l in(args.repairs/'train.jsonl').read_text().splitlines()]
    groups={}
    for row in repairs:groups.setdefault(row['meta']['trajectory_id'],[]).append(row)
    controls=[]
    for identity,rows in groups.items():
        rows.sort(key=lambda r:int(r['id'].rsplit('_step_',1)[1]))
        if len(rows)!=4:raise ValueError('Expected four correct repair actions.')
        folder=args.out/'control/executions'/identity;folder.mkdir(parents=True)
        first=copy.deepcopy(rows[0]['messages'][:2])
        block=next(b for m in first if m['role']=='user'for b in m['content']if b['type']=='image')
        load_observation(block).save(folder/'source.png')
        frozen=initial_messages(folder/'source.png','control snapshot')[1]['content'][0]
        block.clear();block.update(frozen)
        history=first
        for step,row in enumerate(rows):
            target=copy.deepcopy(row['target_message'])
            controls.append({**copy.deepcopy(row),'id':'control_'+row['id'],'messages':copy.deepcopy(history),
                'meta':{**row['meta'],'control':'ordinary correct source history; same target action'}})
            function=target['tool_calls'][0]['function']
            feedback,passed=execute_action(function['name'],function['arguments'],folder,step,True,True,True)
            if not passed:raise ValueError('Control reference failed.')
            history.extend([target,feedback])
        validate_outputs(folder)
        reference=np.load(args.repairs/'executions'/identity/'edited_rgb.npy',allow_pickle=False)
        actual=np.load(folder/'edited_rgb.npy',allow_pickle=False)
        if not np.array_equal(reference,actual):raise ValueError('Control and repair targets differ.')
    if len(controls)!=len(repairs)or any(a['target_message']!=b['target_message']for a,b in zip(repairs,controls)):
        raise ValueError('Supervision/order not matched.')
    hashes={name:write_rows(args.out/name/'train.jsonl',normal+extra)
            for name,extra in [('feedback',repairs),('control',controls)]}
    report={'normal_actions':len(normal),'additional_actions':len(repairs),'rows_per_condition':len(normal)+len(repairs),
        'same_target_actions_and_order':True,'same_reference_rgb':True,'source_photos_unchanged':True,
        'new_teacher_api_calls':0,'dataset_sha256':hashes,'repair_source':str(args.repairs),
        'normal_source_sha256':hashlib.sha256((ROOT/'data/agent_sft_en_27b_37photos_20261125/train.jsonl').read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2))
    print('NATIVE_FEEDBACK_MATCHED',json.dumps(report),flush=True)


if __name__=='__main__':main()
