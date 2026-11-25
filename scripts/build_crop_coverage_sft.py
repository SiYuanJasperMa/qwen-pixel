"""Whole-target crop recovery on existing approved training photographs only."""
import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pixel.agent_protocol import assistant_call,execute_action,initial_messages
from pixel.image_observations import load_observation
from pixel.roi_augmentation import normalized_box
from build_semantic_repair_sft import raster


def coverage(mask,box):
    x0,y0,x1,y1=box
    return float(mask[y0:y1,x0:x1].sum()/mask.sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--photos',type=int,default=12);args=parser.parse_args()
    if args.photos<1:raise ValueError('Need at least one real training source.')
    if args.out.exists():raise ValueError('Use a fresh crop curriculum output.')
    args.out.mkdir(parents=True)
    source=ROOT/'data/agent_sft_en_27b_37photos_20261125/train.jsonl'
    originals=[json.loads(l)for l in source.read_text().splitlines()]
    groups={}
    for row in originals:groups.setdefault(row['meta']['trajectory_id'],[]).append(row)
    selected=[];seen=set()
    for identity,rows in groups.items():
        scene=rows[0]['meta']['original_scene_id']
        if scene in seen:continue
        seen.add(scene);selected.append((identity,rows))
        if len(selected)==args.photos:break
    if len(selected)!=args.photos:raise ValueError('Insufficient distinct approved training sources.')
    output=[];controls=[];audits=[]
    for identity,rows in selected:
        rows.sort(key=lambda r:int(r['id'].rsplit('_step_',1)[1]))
        if len(rows)!=4:raise ValueError('Expected one accepted four-action reference.')
        python=rows[1]
        block=next(b for m in python['messages']if m['role']=='user'for b in m['content']if b['type']=='image')
        source_image=load_observation(block);width,height=source_image.size
        code=python['target_message']['tool_calls'][0]['function']['arguments']['code']
        tree=ast.parse(code)
        polygons=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name)and t.id=='polygons_roi'for t in n.targets))
        metadata=next(json.loads(b['text'])for m in reversed(python['messages'])if m['role']=='tool'
                      and isinstance(m['content'],list)for b in m['content']if b['type']=='text')
        box=metadata['source_bbox_pixels'];x0,y0,x1,y1=box
        mask=raster(polygons,box,(width,height))
        if not mask.any():raise ValueError('Approved source mask is empty.')
        mx,my=(x0+x1)//2,(y0+y1)//2
        partial=[b for b in [[x0,y0,mx,y1],[mx,y0,x1,y1],[x0,y0,x1,my],[x0,my,x1,y1]]
                 if b[2]>b[0]and b[3]>b[1]and coverage(mask,b)<.7]
        if not partial:raise ValueError('Could not construct a genuinely incomplete crop.')
        partial=min(partial,key=lambda b:abs(coverage(mask,b)-.5))
        cw=max(1,min((x1-x0)//2,width//3));ch=max(1,min((y1-y0)//2,height//3))
        distant=[[0,0,cw,ch],[width-cw,0,width,ch],[0,height-ch,cw,height],
                 [width-cw,height-ch,width,height]]
        distant=min(distant,key=lambda b:coverage(mask,b))
        expected_block=next(b for m in reversed(rows[-1]['messages'])if isinstance(m.get('content'),list)
                            for b in m['content']if b['type']=='image')
        expected_rgb=np.asarray(load_observation(expected_block))
        for fault,bad_box in [('partial',partial),('displaced',distant)]:
            if coverage(mask,bad_box)>.7:raise ValueError('Insufficient target omission.')
            name=identity+'_crop_'+fault;folder=args.out/'executions'/name;folder.mkdir(parents=True)
            source_image.save(folder/'source.png')
            frozen=initial_messages(folder/'source.png','snapshot only')[1]['content'][0]
            history=copy.deepcopy(rows[0]['messages'])
            for m in history:
                if m['role']=='user':
                    for b in m['content']:
                        if b['type']=='image':b.clear();b.update(frozen)
            bad=assistant_call('view_image',{'path':'source.png','bbox_2d':normalized_box(bad_box,(width,height))})
            feedback,passed=execute_action('view_image',bad['tool_calls'][0]['function']['arguments'],folder,0,True,True,True)
            if not passed:raise ValueError('Constructed crop failed actual execution.')
            actual_box=json.loads((folder/'region.json').read_text())['source_bbox_pixels']
            if actual_box!=bad_box:raise ValueError('Observed crop differs from the measured source box.')
            history.extend([bad,feedback])
            for step,row in enumerate(rows,1):
                target=copy.deepcopy(row['target_message'])
                output.append({'id':name+'_step_'+str(step),'task':'agent_edit','messages':copy.deepcopy(history),
                    'target_message':target,'tools':copy.deepcopy(row['tools']),
                    'meta':{**row['meta'],'trajectory_id':name,'curriculum':'whole_target_crop_recovery',
                            'fault':fault,'bad_crop_not_supervised':True}})
                control=copy.deepcopy(row);control['id']='control_'+name+'_step_'+str(step)
                controls.append(control)
                f=target['tool_calls'][0]['function']
                feedback,passed=execute_action(f['name'],f['arguments'],folder,step,True,True,True)
                if not passed:raise ValueError('Crop recovery reference failed actual execution.')
                history.extend([target,feedback])
            actual=np.load(folder/'edited_rgb.npy',allow_pickle=False)
            if not np.array_equal(actual,expected_rgb):raise ValueError('Recovery changed frozen target RGB.')
            audits.append({'id':name,'scene':rows[0]['meta']['original_scene_id'],'bad_box':bad_box,
                'bad_crop_target_coverage':coverage(mask,bad_box),'reference_rgb_exact':True,
                'bad_action_supervised':False})
    if any(a['target_message']!=b['target_message']for a,b in zip(output,controls)):
        raise ValueError('Crop/control target supervision differs.')
    for name,rows in [('crop_recovery',output),('ordinary_control',controls)]:
        folder=args.out/name;folder.mkdir()
        (folder/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    report={'photos':len(selected),'rows_per_condition':len(output),'same_target_actions_and_order':True,
        'all_final_rgb_exact':True,'new_teacher_api_calls':0,'validation_images_used':0,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':audits}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2))
    print('CROP_COVERAGE_REFERENCE_AUDIT',json.dumps({k:v for k,v in report.items()if k!='cases'}),flush=True)


if __name__=='__main__':main()
