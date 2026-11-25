"""Small paired crop-feedback curriculum from existing verified training photos."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import assistant_call,execute_action,initial_messages
from pixel.image_observations import load_observation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise ValueError('Use a fresh curriculum directory.')
    args.out.mkdir(parents=True)
    original=[json.loads(s)for s in (ROOT/'data/agent_sft_en_27b_37photos_20261125/train.jsonl').read_text().splitlines()]
    probes=[json.loads(s)for s in (ROOT/'outputs/agent_sft_en_27b_crop_recovery_probe_20261125/prefixes.jsonl').read_text().splitlines()]
    negative=[r for r in probes if r['id'].endswith('_partial_roi')]
    output=[];manifest=[]
    for index,probe in enumerate(negative):
        identity=probe['meta']['trajectory_id']
        reference_rows=[r for r in original if r['meta']['trajectory_id']==identity]
        assert len(reference_rows)==4
        reference_rows.sort(key=lambda r:int(r['id'].rsplit('_step_',1)[1]))
        for r in reference_rows:
            row=copy.deepcopy(r);row['id']='positive_'+row['id'];row['meta']['curriculum']='verified_complete_crop';output.append(row)
        folder=args.out/'executions'/identity;folder.mkdir(parents=True)
        source_block=next(b for m in probe['messages']if m['role']=='user'for b in m['content']if b['type']=='image')
        load_observation(source_block).save(folder/'source.png')
        frozen=initial_messages(folder/'source.png','capture actual source')[1]['content'][0]
        history=copy.deepcopy(probe['messages'][:2])
        for m in history:
            if m['role']=='user':
                for b in m['content']:
                    if b['type']=='image':b.clear();b.update(frozen)
        bad=probe['messages'][-2]['tool_calls'][0]['function']['arguments']
        action=assistant_call('view_image',bad);feedback,passed=execute_action('view_image',bad,folder,0,True,True,True)
        assert passed;history.extend([action,feedback])
        for step,reference in enumerate(reference_rows,1):
            target=copy.deepcopy(reference['target_message'])
            output.append({'id':identity+'_recovery_step_'+str(step),'task':'agent_edit','messages':copy.deepcopy(history),
                'target_message':target,'tools':copy.deepcopy(reference['tools']),
                'meta':{'trajectory_id':identity+'_recovery','original_scene_id':probe['meta']['original_scene_id'],
                    'curriculum':'actual_partial_crop_then_correct_crop','source':'annotation-derived crop feedback supervision; not GPT reasoning'}})
            f=target['tool_calls'][0]['function']
            feedback,passed=execute_action(f['name'],f['arguments'],folder,step,True,True,True)
            if not passed:raise ValueError('Recovery reference failed: '+json.dumps(feedback))
            history.extend([target,feedback])
        final=reference_rows[-1]
        block=next(b for m in reversed(final['messages'])if isinstance(m.get('content'),list)for b in m['content']if b.get('type')=='image')
        expected=np.asarray(load_observation(block));actual=np.load(folder/'edited_rgb.npy',allow_pickle=False)
        if not np.array_equal(expected,actual):raise ValueError('Recovery program changed verified reference RGB.')
        (folder/'trajectory.json').write_text(json.dumps({'messages':history,'complete':True,'teacher_rgb_exact':True},indent=2))
        manifest.append({'trajectory_id':identity,'original_scene_id':probe['meta']['original_scene_id'],
            'reference_rgb_exact':True,'bad_initial_action_not_supervised':True})
    assert len(output)==32 and len(manifest)==4
    (args.out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in output))
    (args.out/'manifest.json').write_text(json.dumps({'rows':len(output),'photos':len(manifest),
        'source_training_data_unchanged':True,'new_teacher_api_calls':0,'references':manifest},indent=2))
    print('PAIRED_CROP_CORRECTION_SFT',len(output),flush=True)


if __name__=='__main__':main()
