"""Compare actually generated recovery histories with their SFT code prefixes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'train'))
from run_pixel_sft import encode


def hashes(inputs):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype),
        'sha256':hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for k,v in inputs.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    adapter=ROOT/'outputs/agent_sft_crop_feedback_paired_20261125/train/checkpoint-480'
    processor=AutoProcessor.from_pretrained(adapter)
    processor.image_processor.size={'shortest_edge':262144,'longest_edge':262144}
    rows=[json.loads(l)for l in(ROOT/'data/sft_crop_correction_20261125_worker/train.jsonl').read_text().splitlines()]
    cases=[]
    for row in rows:
        if not row['id'].endswith('_recovery_step_2'):continue
        parent=row['meta']['trajectory_id'].removesuffix('_recovery')
        path=ROOT/'outputs/agent_sft_crop_full_recovery_paired_20261125'/f'{parent}_step_1_partial_roi'/'actor/trajectory.json'
        trajectory=json.loads(path.read_text())
        native_turn=next(t for t in trajectory['turns']if t['action']=='python')
        assistant_indexes=[i for i,m in enumerate(trajectory['messages'])if m['role']=='assistant']
        # One restored historical crop, then the model's generated corrected crop.
        actual_messages=trajectory['messages'][:assistant_indexes[2]]
        teacher=encode(processor,row,'cpu')
        generated=encode(processor,{'messages':actual_messages,'tools':row['tools']},'cpu')
        if hashes(generated)!=native_turn['input_fingerprints']:
            raise ValueError('Reconstructed history differs from actual actor input.')
        fields={k:torch.equal(teacher[k],generated[k])for k in teacher}
        case={'id':row['id'],'actual_actor_input_reproduced':True,'fields_equal':fields,
              'teacher_code_prefix_equal':all(fields.values())}
        if not fields['input_ids']:
            a,b=teacher['input_ids'][0].tolist(),generated['input_ids'][0].tolist()
            i=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),min(len(a),len(b)))
            case['first_text_difference']={'position':i,
                'teacher':processor.tokenizer.decode(a[max(0,i-25):i+50]),
                'actor':processor.tokenizer.decode(b[max(0,i-25):i+50])}
        cases.append(case)
    args.out.write_text(json.dumps(cases,indent=2))
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
