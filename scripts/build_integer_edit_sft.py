"""Re-execute the same accepted native edits with shorter integer ROI targets."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import assistant_call,copy_source_image,execute_action
from pixel.integer_edit_protocol import initial,integer_polygons,program,TOOLS


def convert(task,data,out):
    reference=data/'executions'/task['id']
    original=json.loads((reference/'trajectory.json').read_text())
    tree=ast.parse(original['messages'][4]['tool_calls'][0]['function']['arguments']['code'])
    polygons=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='polygons_roi'for t in n.targets))
    box=json.loads((reference/'region.json').read_text())['source_bbox_pixels']
    integers=integer_polygons(polygons,box)
    workspace=out/'executions'/task['id'];workspace.mkdir(parents=True,exist_ok=True)
    copy_source_image(reference/'source.png',workspace/'source.png');messages=initial(workspace/'source.png',task['instruction'])
    sequence=[original['messages'][2],assistant_call('python',{'code':program(integers,task['operation'],task.get('tint_color')or'red')}),
        assistant_call('view_image',{'path':'edited.png'}),assistant_call('finish',{'image_path':'edited.png','rgb_path':'edited_rgb.npy'})]
    rows=[]
    for step,action in enumerate(sequence):
        function=action['tool_calls'][0]['function'];prefix=json.loads(json.dumps(messages))
        for attempt in range(3):
            feedback,passed=execute_action(function['name'],function['arguments'],workspace,step,True,True,True)
            if passed or 'TimeoutExpired'not in str(feedback):break
            with (workspace/'construction_retries.jsonl').open('a')as file:file.write(json.dumps({'step':step,'attempt':attempt,'feedback':feedback,'model_visible':False})+'\n')
        if not passed:raise ValueError('Integer reference execution failed: '+task['id']+' '+str(feedback))
        rows.append({'id':task['id']+f'_step_{step}','task':'agent_edit','messages':prefix,
            'target_message':action,'tools':TOOLS,'meta':{'trajectory_id':task['id'],
                'original_scene_id':task['original_scene_id'],'geometry_format':'integer_crop_0_1000',
                'source':'Coordinate-only conversion of accepted annotation-derived RGB programs'}})
        messages.extend([action,feedback])
    if not np.array_equal(np.load(reference/'edited_rgb.npy'),np.load(workspace/'edited_rgb.npy')):
        raise ValueError('Representation conversion changed native RGB: '+task['id'])
    (workspace/'trajectory.json').write_text(json.dumps({'instruction':task['instruction'],'messages':messages,
        'native_reference_rgb_exact':True},indent=2))
    converted={**task,'image':str(workspace/'source.png'),'target_bbox_pixels':box}
    print('INTEGER_NATIVE_RGB_EXACT',task['id'],flush=True);return rows,converted


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'train.jsonl').exists():raise ValueError('Use a fresh integer representation variant.')
    original=[json.loads(l)for l in(args.data/'train_verified.jsonl').open()]
    approved={r['meta']['trajectory_id']for r in original}
    tasks=[t for t in(json.loads(l)for l in(args.data/'tasks.jsonl').open())if t['id']in approved]
    if args.limit:tasks=tasks[:args.limit]
    rows=[];converted=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
        for built,task in pool.map(lambda task:convert(task,args.data,args.out),tasks):rows.extend(built);converted.append(task)
    (args.out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    (args.out/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in converted))
    shutil.copyfile(args.data/'development_tasks.jsonl',args.out/'development_tasks.jsonl')
    manifest={'native_training_images':len({t['original_scene_id']for t in converted}),'trajectories':len(converted),
        'assistant_actions':len(rows),'all_reference_RGB_exact':True,'same_reference_edit_semantics':True,
        'geometry_format':'integer_crop_0_1000','source_data_sha256':hashlib.sha256((args.data/'train_verified.jsonl').read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256((args.out/'train.jsonl').read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('INTEGER_EDIT_DATA_READY',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
