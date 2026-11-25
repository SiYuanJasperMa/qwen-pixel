"""Persist and validate the model-visible agent interface used for SFT."""
import hashlib
import json
import re
import copy
from pathlib import Path


def user_template(messages):
    user=next((m for m in messages if m['role']=='user'),None)
    if user is None:raise ValueError('SFT contract requires an initial user input.')
    content=user['content']
    blocks=content if isinstance(content,list)else[{'type':'text','text':content}]
    normalized=[]
    for block in blocks:
        if block.get('type')=='image':normalized.append({'type':'image'})
        elif block.get('type')=='text':
            text=block['text']
            instruction=user.get('edit_instruction')
            if instruction is not None:text=text.replace('EDIT: '+instruction,'EDIT: {instruction}',1)
            else:text=re.sub(r'(?m)^EDIT: .*','EDIT: {instruction}',text)
            text=re.sub(r'width=\d+,\s*height=\d+','width={width}, height={height}',text)
            normalized.append({'type':'text','text':text})
    return normalized


def infer_runtime_features(system,tools):
    functions=[t.get('function',{})for t in tools]
    description=next((f.get('description','')for f in functions if f.get('name')=='python'),'')
    rgb='emit_rgb'in system or'emit_rgb'in description
    roi=any(f.get('name')=='view_image'and'bbox_2d'in f.get('parameters',{}).get('properties',{})for f in functions)
    checked=rgb and('rejects'in description or'rejects out-of-range'in system or
                   ('validates'in description and'rounds/clips'not in system))
    return {'rgb':rgb,'roi':roi,'checked_rgb':checked,'python_scope':'per_call','files_persist':True}


def signature(system, tools, mode, template=None,runtime_features=None):
    value={'system':system, 'tools':tools, 'tool_image_mode':mode}
    if template is not None:value['user_prompt_template']=template
    if runtime_features is not None:value['runtime_features']=runtime_features
    raw = json.dumps(value, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def training_contract(rows, image_pixels):
    agent_rows = [r for r in rows if 'messages' in r]
    if not agent_rows:
        return None
    contracts = {}
    for row in agent_rows:
        target=row.get('target_message')
        if target is not None:
            if target.get('role')!='assistant':
                raise ValueError('Agent SFT target must be an assistant action: '+row.get('id','unknown'))
            if len(target.get('tool_calls',[]))!=1:
                raise ValueError('Agent SFT requires one complete tool action per target: '+row.get('id','unknown'))
        system = row['messages'][0]['content']
        tools = row.get('tools', [])
        mode = row.get('tool_image_mode', 'tool')
        template=user_template(row['messages'])
        features=row.get('runtime_features')or infer_runtime_features(system,tools)
        key = signature(system, tools, mode,template,features)
        contracts[key] = {'system':system,'tools':tools,'tool_image_mode':mode,
                          'user_prompt_template':template,'runtime_features':features}
    if len(contracts) != 1:
        raise ValueError('Mixed agent system/tool/image contracts: split them before SFT.')
    digest, contract = next(iter(contracts.items()))
    return {**contract,'signature':digest,'image_max_pixels':image_pixels,
            'version':'qwen-pixel-sft-contract-v3'}


def validate_runtime_contract(adapter, system, tools, mode, image_pixels,messages=None,runtime_features=None):
    if not adapter:
        return
    path = Path(adapter) / 'agent_contract.json'
    if not path.exists():
        return  # Older checkpoints predate the contract audit.
    expected = json.loads(path.read_text())
    if expected.get('user_prompt_template')is not None and messages is None:
        raise ValueError('Initial user messages are required to verify the SFT runtime contract.')
    template=user_template(messages)if expected.get('user_prompt_template')is not None else None
    actual=runtime_features or infer_runtime_features(system,tools)
    required=expected.get('runtime_features')
    if required is not None and actual!=required:
        raise ValueError('Inference execution behavior differs from the saved SFT runtime features.')
    if signature(system, tools, mode,template,required) != expected['signature']:
        raise ValueError('Inference system/tools/tool-image mode differ from the saved SFT contract.')
    if image_pixels != expected['image_max_pixels']:
        raise ValueError('Inference image token budget differs from the saved SFT contract.')


def checkpoint_interface(adapter,messages,tools,runtime_features=None):
    """Explicitly restore a checkpoint's taught text, preserving runtime schemas."""
    path=Path(adapter)/'agent_contract.json'
    if not path.is_file():raise ValueError('Checkpoint has no saved agent interface.')
    expected=json.loads(path.read_text())
    def schemas(items):
        return [(t['function']['name'],t['function']['parameters'])for t in items]
    if schemas(tools)!=schemas(expected['tools']):
        raise ValueError('Saved tool arguments require a different runtime profile.')
    actual=runtime_features or infer_runtime_features(messages[0]['content'],tools)
    required=expected.get('runtime_features')or infer_runtime_features(expected['system'],expected['tools'])
    if actual!=required:
        raise ValueError('Saved RGB numeric validation or crop behavior requires a different runtime profile.')
    restored=copy.deepcopy(messages);restored[0]['content']=expected['system']
    return restored,copy.deepcopy(expected['tools'])
