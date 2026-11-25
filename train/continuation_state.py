"""Exact optimizer/sampler/RNG continuation bound to checkpoint and data identity."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

VERSION='qwen_pixel_training_state_v1'


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save_latest(path,checkpoint,optimizer,identity,param_names,step,index,cursor,tokens,elapsed):
    checkpoint=Path(checkpoint).resolve()
    state={'version':VERSION,'checkpoint':str(checkpoint),'weights_sha256':file_hash(checkpoint/'adapter_model.safetensors'),
        'identity':identity,'param_names':param_names,'step':step,'index':list(index),'cursor':cursor,
        'supervised_tokens':tokens,'elapsed_seconds':elapsed,'optimizer':optimizer.state_dict(),
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else None}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    try:
        torch.save(state,temp);temp.replace(path)
    finally:
        if temp.exists():temp.unlink()


def read_state(path,identity=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=VERSION:raise ValueError('Unsupported training continuation state.')
    checkpoint=Path(state['checkpoint'])
    if file_hash(checkpoint/'adapter_model.safetensors')!=state['weights_sha256']:
        raise ValueError('Saved training state does not match checkpoint weights.')
    if identity is not None and identity!=state['identity']:
        raise ValueError('Changed data, model, label policy or training settings; not an exact continuation.')
    index=state['index']
    if sorted(index)!=list(range(len(index)))or not 0<=state['cursor']<=len(index):
        raise ValueError('Invalid saved sampler position.')
    return state


def restore(state,optimizer,param_names):
    if param_names!=state['param_names']:raise ValueError('Optimizer parameter order changed.')
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']is not None:
        if len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG topology changed.')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'],list(state['index']),state['cursor'],state['supervised_tokens'],state['elapsed_seconds']
