import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from trajectory_io import encode_trajectory
from pixel.agent_protocol import assistant_call
from pixel.clean_edit_protocol import TOOLS


@pytest.fixture(scope='module')
def processor():
    p=AutoProcessor.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    p.image_processor.size={'shortest_edge':1024,'longest_edge':1024}
    return p


def row(tmp_path,code='print(1)'):
    image=tmp_path/'source.png';Image.new('RGB',(32,32),'green').save(image)
    return {'id':'actual_native_stop','messages':[{'role':'system','content':'Use Python to edit RGB.'},
        {'role':'user','content':[{'type':'image','image':str(image)},
                                 {'type':'text','text':'Brighten the image.'}]}],
        'tools':TOOLS,'target_message':assistant_call('python',{'code':code})}


def test_only_assistant_through_eos_is_supervised_with_image_prefix_unchanged(processor,tmp_path):
    data=row(tmp_path)
    prompt=encode_trajectory(processor,data,'cpu')
    full=encode_trajectory(processor,data,'cpu',supervised=True)
    n=prompt['input_ids'].shape[1];stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    end=n+int(torch.nonzero(full['input_ids'][0,n:]==stop)[0])+1
    assert torch.equal(full['input_ids'][:,:n],prompt['input_ids'])
    assert torch.equal(full['pixel_values'],prompt['pixel_values'])
    assert torch.all(full['labels'][:,:n]==-100)
    assert torch.all(full['labels'][:,n:end]!=-100)
    assert torch.all(full['labels'][:,end:]==-100)
    assert full['input_ids'].shape[1]>end  # Real template's separator exists.


def test_stop_inside_code_cannot_become_a_silent_partial_target(processor,tmp_path):
    with pytest.raises(ValueError,match='before the complete target'):
        encode_trajectory(processor,row(tmp_path,"print('<|im_end|>')"),'cpu',supervised=True)


def test_exact_sampled_token_branch_preserves_unstopped_samples(processor,tmp_path):
    data=row(tmp_path);data.pop('target_message')
    sampled=processor.tokenizer.encode('partial output',add_special_tokens=False)
    data['target_token_ids']=sampled
    prompt=encode_trajectory(processor,data,'cpu')
    full=encode_trajectory(processor,data,'cpu',supervised=True)
    n=prompt['input_ids'].shape[1]
    assert full['input_ids'][0,n:].tolist()==sampled
    assert full['labels'][0,n:].tolist()==sampled


@pytest.mark.parametrize('declared_target',[{'role':'user','content':'not an assistant'},{}])
def test_sampled_tokens_cannot_bypass_declared_assistant_role(processor,tmp_path,declared_target):
    data=row(tmp_path);data['target_message']=declared_target
    data['target_token_ids']=processor.tokenizer.encode('an action',add_special_tokens=False)
    with pytest.raises(ValueError,match='Only an assistant'):
        encode_trajectory(processor,data,'cpu',supervised=True)
