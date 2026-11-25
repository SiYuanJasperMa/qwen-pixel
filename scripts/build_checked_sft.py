"""Replay existing teachers with numeric checks; add actual error/repair contexts.

No new teacher API calls or held-out images are used. Canonicalization only
makes the serializer's existing clipping explicit; output pixels must match.
"""
import ast
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import (CHECKED_RGB_SYSTEM, CHECKED_RGB_TOOLS,NORMALIZED_RGB_SYSTEM,
    REGION_EDIT_SYSTEM,REGION_EDIT_TOOLS,add_region_guidance,
    assistant_call, copy_source_image, execute_action, initial_messages)


class ClipEmitter(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'emit_rgb' and node.args:
            node.args[0] = ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),
                attr='clip',ctx=ast.Load()),args=[node.args[0],ast.Constant(0),ast.Constant(255)],keywords=[])
        return node


def canonical_code(code):
    try:
        return ast.unparse(ast.fix_missing_locations(ClipEmitter().visit(ast.parse(code))))
    except SyntaxError:
        return code


def append_row(rows, messages, action, identity, step, provenance,tools=None):
    rows.append({'id':f'{identity}_{step:02d}','task':'agent_edit','messages':copy.deepcopy(messages),
        'target_message':copy.deepcopy(action),'tools':tools or CHECKED_RGB_TOOLS,
        'meta':{'trajectory_id':identity,'provenance':provenance}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--normalized-coordinates',action='store_true')
    parser.add_argument('--roi-first',action='store_true')
    args=parser.parse_args()
    dataset=ROOT/'data'/('sft_roi_v1'if args.roi_first else'sft_normalized_v1'if args.normalized_coordinates else'sft_checked_v1')
    system=REGION_EDIT_SYSTEM if args.roi_first else NORMALIZED_RGB_SYSTEM if args.normalized_coordinates else CHECKED_RGB_SYSTEM
    tools=REGION_EDIT_TOOLS if args.roi_first else CHECKED_RGB_TOOLS
    dataset.mkdir(parents=True,exist_ok=True)
    source_rows=[json.loads(l)for l in(ROOT/'data/sft_seven_tasks_v1/train.jsonl').open()]
    tasks=[json.loads(l)for l in(ROOT/'data/sft_seven_tasks_v1/tasks.jsonl').open()]
    task_map={t['id']:t for t in tasks}
    groups={}
    for row in source_rows:groups.setdefault(row['meta']['trajectory_id'],[]).append(row)
    rows=[];verified=[];repairs=0
    for identity,samples in groups.items():
        workspace=dataset/'executions'/identity
        workspace.mkdir(parents=True,exist_ok=True)
        copy_source_image(task_map[identity]['image'],workspace/'source.png')
        messages=initial_messages(workspace/'source.png',task_map[identity]['instruction'])
        messages[0]['content']=system
        if args.roi_first:add_region_guidance(messages)
        with Image.open(workspace/'source.png')as image:width,height=image.size
        teacher=np.asarray(Image.open(ROOT/'data/agent_rgb_teacher_v6'/identity/'edited.png').convert('RGB'))
        frame=None
        offset=0
        if args.roi_first:
            source=np.asarray(Image.open(workspace/'source.png').convert('RGB'))
            yy,xx=np.where(np.any(teacher!=source,axis=-1))
            if not len(xx):raise ValueError('Teacher edit has no changed pixels.')
            frame=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            normalized=[round(value*1000/(width if i%2==0 else height))for i,value in enumerate(frame)]
            mapped=[round(value*(width if i%2==0 else height)/1000)for i,value in enumerate(normalized)]
            if mapped!=frame:
                normalized=[round(value*1000/(width if i%2==0 else height),8)for i,value in enumerate(frame)]
            region=assistant_call('view_image',{'path':'source.png','bbox_2d':normalized})
            feedback,passed=execute_action('view_image',region['tool_calls'][0]['function']['arguments'],
                workspace,0,True,True,True)
            if not passed:raise ValueError('ROI read failed.')
            actual_frame=json.loads(feedback['content'][0]['text'])['source_bbox_pixels']
            if actual_frame!=frame:raise ValueError('ROI coordinates do not map to teacher pixels.')
            append_row(rows,messages,region,identity,0,
                'derived_ROI_supervision_from_actual_accepted_teacher_pixel_changes; actual_region_reader_feedback; box_must_be_predicted_by_student_at_inference',tools)
            messages.extend([region,feedback]);offset=1
        for step,row in enumerate(samples):
            action=copy.deepcopy(row['target_message'])
            call=action['tool_calls'][0]['function']
            if args.roi_first and call['name']=='view_image'and call['arguments'].get('path')=='source.png':
                continue
            if call['name']=='python':
                code=canonical_code(call['arguments']['code'])
                if args.normalized_coordinates or args.roi_first:
                    from pixel.coordinate_sft import normalize_drawing_code
                    code,_=normalize_drawing_code(code,width,height,frame)
                call['arguments']['code']=code
            feedback,passed=execute_action(call['name'],call['arguments'],workspace,step+offset,True,args.roi_first,True)
            if passed:
                append_row(rows,messages,action,identity,step+offset,
                    'existing_API_teacher_action_canonicalized_and_replayed_with_checked_RGB; failed_action_targets_excluded',tools)
            messages.extend([action,feedback])
        final=np.asarray(Image.open(workspace/'edited.png').convert('RGB'))
        if not np.array_equal(final,teacher):raise ValueError('Canonicalized pixels differ: '+identity)
        verified.append(identity)
        (workspace/'trajectory.json').write_text(json.dumps({'messages':messages,'pixel_equal_teacher':True},indent=2))

        # Corrupt an alpha computation on a training image; supervise only the
        # correct response after actual numeric feedback, never the bad action.
        repair_id=identity+'_numeric_repair'
        repair_dir=dataset/'executions'/repair_id
        repair_dir.mkdir(exist_ok=True)
        copy_source_image(task_map[identity]['image'],repair_dir/'source.png')
        repair_messages=initial_messages(repair_dir/'source.png',task_map[identity]['instruction'])
        repair_messages[0]['content']=system
        if args.roi_first:add_region_guidance(repair_messages)
        bad_code='''arr=pixels.astype(np.float32)
h,w=arr.shape[:2]
alpha=np.zeros((h,w),dtype=np.float32)
alpha[h//4:3*h//4,w//4:3*w//4]=200
edited=np.minimum(arr+30,255)
out=arr*(1-alpha[...,None])+edited*alpha[...,None]
emit_rgb(out)'''
        bad=assistant_call('python',{'code':bad_code})
        feedback,passed=execute_action('python',{'code':bad_code},repair_dir,0,True,args.roi_first,True)
        if passed:raise ValueError('Fault injection did not produce real numeric failure.')
        repair_messages.extend([bad,feedback])
        # Preserve all genuine teacher revisions. The first emitted candidate
        # may differ from the final accepted edit; do not replace its trajectory
        # with a single early program and label that as a successful repair.
        sequence=[copy.deepcopy(m) for m in messages
                  if m['role']=='assistant' and m.get('tool_calls')]
        for step,action in enumerate(sequence,1):
            call=action['tool_calls'][0]['function']
            result,passed=execute_action(call['name'],call['arguments'],repair_dir,step,True,args.roi_first,True)
            if passed:
                append_row(rows,repair_messages,action,repair_id,step,
                    'synthetic_alpha_fault_on_training_image; actual_error_feedback; repair_replays_the_complete_validated_teacher_revision_sequence',tools)
            repair_messages.extend([action,result])
        if not np.array_equal(np.load(repair_dir/'edited_rgb.npy'),teacher):
            raise ValueError('Repair did not reproduce teacher pixels: '+identity)
        repairs+=1
        (repair_dir/'trajectory.json').write_text(json.dumps({'messages':repair_messages,
            'bad_action_supervised':False,'pixel_equal_teacher':True},indent=2))
    (dataset/'train.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
    (dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    summary={'verified_teacher_trajectories':verified,'repair_trajectories':repairs,
        'assistant_actions':len(rows),'bad_action_targets_supervised':0,
        'teacher_pixels_exact':True,'heldout_images_used_in_training':False,
        'normalized_coordinates':args.normalized_coordinates}
    summary['roi_first']=args.roi_first
    if args.roi_first:
        summary['roi_supervision']='Box derived from accepted teacher output only as a training target. Student must predict the box from source and instruction; reader supplies no object detector or oracle box.'
    (dataset/'manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
