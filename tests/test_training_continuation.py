import random

import numpy as np
import pytest
import torch
from safetensors.torch import save_file,load_file

from train.continuation_state import save_latest,read_state,restore


def test_interrupted_training_restores_optimizer_sampling_and_all_cpu_rng(tmp_path):
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.02)
    names=[name for name,_ in model.named_parameters()]
    def update():
        sample=random.randrange(4);noise=float(np.random.random());x=torch.randn(2)+sample
        loss=(model(x)-noise).square().sum();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        return sample,noise,float(loss.detach())
    for _ in range(3):update()
    checkpoint=tmp_path/'checkpoint-3';checkpoint.mkdir()
    save_file(model.state_dict(),checkpoint/'adapter_model.safetensors')
    state_path=tmp_path/'latest.pt';identity={'data':'same','label_policy':'nativeEOS'}
    save_latest(state_path,checkpoint,optimizer,identity,names,3,[2,0,3,1],2,90,4.0)
    expected=[update()for _ in range(4)]
    expected_weights={k:v.clone()for k,v in model.state_dict().items()}
    state=read_state(state_path,identity)
    resumed=torch.nn.Linear(2,1);resumed.load_state_dict(load_file(checkpoint/'adapter_model.safetensors'))
    resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.02)
    values=restore(state,resumed_optimizer,names)
    assert values==(3,[2,0,3,1],2,90,4.0)
    model,optimizer=resumed,resumed_optimizer
    actual=[update()for _ in range(4)]
    assert expected==actual
    assert all(torch.equal(expected_weights[k],v)for k,v in resumed.state_dict().items())
    with pytest.raises(ValueError,match='Changed data'):
        read_state(state_path,{'data':'different'})
    (checkpoint/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint weights'):
        read_state(state_path,identity)
