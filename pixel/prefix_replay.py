"""Restore only observed history, verifying it against fresh real execution."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .agent_protocol import execute_action, parse_call,initial_messages
from .image_observations import load_observation


def action(message):
    calls=message.get('tool_calls')
    if calls:
        if len(calls)!=1:raise ValueError('History requires exactly one action.')
        f=calls[0]['function']
        return f['name'],f['arguments']
    return parse_call(message['content'])


def feedback_signature(message):
    parts=message['content']
    parts=parts if isinstance(parts,list)else[{'type':'text','text':parts}]
    result=[]
    for part in parts:
        if part['type']=='image':
            image=load_observation(part)
            result.append(('image',image.size,image.tobytes()))
        elif part['type']=='text':
            text=part['text']
            try:text=json.loads(text)
            except json.JSONDecodeError:pass
            result.append(('text',text))
        else:raise ValueError('Unexpected observed feedback content.')
    return message.get('name'),result


def replay_prefix(row,workspace):
    """Future targets are deliberately not read or executed."""
    workspace=Path(workspace)
    messages=row['messages']
    blocks=[b for m in messages if isinstance(m.get('content'),list)
            for b in m['content']if b.get('type')=='image']
    if not blocks:raise ValueError('Missing initial source observation.')
    for block in blocks:load_observation(block)
    with Image.open(workspace/'source.png')as image:
        source=np.asarray(image.convert('RGB'))
    if not np.array_equal(source,np.asarray(load_observation(blocks[0]))):
        raise ValueError('Runtime source differs from visible initial pixels.')
    rebound=copy.deepcopy(messages)
    source_snapshot=next(b for b in initial_messages(workspace/'source.png','prefix snapshot')[1]['content']
                         if b['type']=='image')
    first_user=next(m for m in rebound if m['role']=='user')
    for index,block in enumerate(first_user['content']):
        if block.get('type')=='image':
            first_user['content'][index]=copy.deepcopy(source_snapshot)
            break
    steps=0
    for index,message in enumerate(messages):
        if message['role']!='assistant':continue
        if index+1>=len(messages)or messages[index+1]['role']!='tool':
            raise ValueError('Historical action is missing its actual tool result.')
        name,args=action(message)
        observed=messages[index+1]
        feedback,passed=execute_action(name,args,workspace,steps,True,True,True)
        if not passed:raise ValueError('Historical action failed in fresh runtime.')
        if feedback_signature(observed)!=feedback_signature(feedback):
            raise ValueError('Visible history differs from freshly executed feedback.')
        # Keep original text exactly, while binding image observations to this
        # fresh actor workspace. Historical evaluator artifact paths stay hidden.
        if isinstance(rebound[index+1]['content'],list):
            actual_images=iter(b for b in feedback['content']if b['type']=='image')
            for j,block in enumerate(rebound[index+1]['content']):
                if block['type']=='image':rebound[index+1]['content'][j]=copy.deepcopy(next(actual_images))
        steps+=1
    return {'historical_actions_reexecuted':steps,'observed_images_verified':len(blocks),
            'source_rgb_equal':True,'all_feedback_equal':True,'future_target_used':False,
            'rebound_messages':rebound}
