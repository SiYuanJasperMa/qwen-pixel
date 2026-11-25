import copy
import hashlib
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_visible_supervision import audit
from pixel.agent_protocol import assistant_call
from pixel.clean_edit_protocol import TOOLS


def test_equivalent_structured_and_raw_native_targets_group_identically(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    image = tmp_path/'source.png'
    Image.new('RGB',(32,32),'green').save(image)
    messages = [{'role':'system','content':'Use Python.'}, {'role':'user','content':[
        {'type':'image','image':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()},
        {'type':'text','text':'Edit this image.'}]}]
    target = assistant_call('python',{'code':'print(1)'})
    kwargs = dict(tools=TOOLS,tokenize=False,enable_thinking=False)
    prompt = tokenizer.apply_chat_template(messages,add_generation_prompt=True,**kwargs)
    full = tokenizer.apply_chat_template(messages+[target],add_generation_prompt=False,**kwargs)
    raw = full[len(prompt):].split('<|im_end|>')[0]
    rows = [{'id':'structured','messages':messages,'tools':TOOLS,'target_message':target},
            {'id':'native','messages':copy.deepcopy(messages),'tools':TOOLS,
             'target_message':{'role':'assistant','content':raw}}]
    result = audit(rows,tokenizer)
    assert result['unique_visible_histories'] == 1
    assert result['divergent_supervision_groups'] == []
    assert result['actions'] == {'python':2}
    # A different sampled action must not disappear merely because the stored
    # target-message text was unchanged.
    rows[-1]['target_token_ids'] = tokenizer.encode('different action<|im_end|>',add_special_tokens=False)
    assert len(audit(rows,tokenizer)['divergent_supervision_groups']) == 1
