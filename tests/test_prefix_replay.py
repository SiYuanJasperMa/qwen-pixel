import copy
import json

import numpy as np
from PIL import Image

from pixel.agent_protocol import assistant_call,execute_action
from pixel.clean_edit_protocol import initial,TOOLS
from pixel.prefix_replay import replay_prefix


def fixture(tmp_path):
    original=tmp_path/'original';original.mkdir()
    im=Image.fromarray(np.arange(8*6*3,dtype=np.uint8).reshape(6,8,3));im.save(original/'source.png')
    messages=initial(original/'source.png','Blur the object.')
    call=assistant_call('view_image',{'path':'source.png','bbox_2d':[0,0,500,1000]})
    feedback,passed=execute_action('view_image',call['tool_calls'][0]['function']['arguments'],original,0,True,True,True)
    assert passed;messages.extend([call,feedback])
    fresh=tmp_path/'fresh';fresh.mkdir();im.save(fresh/'source.png')
    return {'messages':messages,'tools':TOOLS,'target_message':{'role':'assistant','content':'POISON FUTURE'}},fresh


def test_replays_actual_crop_without_using_future_target(tmp_path):
    row,workspace=fixture(tmp_path)
    original=copy.deepcopy(row)
    result=replay_prefix(row,workspace)
    assert result['all_feedback_equal']and result['historical_actions_reexecuted']==1
    assert not result['future_target_used']
    assert row==original
    assert all(str(workspace)in b['image']for m in result['rebound_messages']
               if isinstance(m.get('content'),list)for b in m['content']if b['type']=='image')
    assert json.loads((workspace/'region.json').read_text())['source_bbox_pixels']==[0,0,4,6]


def test_rejects_mutated_feedback_even_with_unchanged_image(tmp_path):
    row,workspace=fixture(tmp_path);row=copy.deepcopy(row)
    content=row['messages'][-1]['content'][0]
    data=json.loads(content['text']);data['source_bbox_pixels'][0]=1;content['text']=json.dumps(data)
    try:replay_prefix(row,workspace)
    except ValueError as e:assert 'feedback' in str(e)
    else:raise AssertionError('Incorrect visible metadata accepted')


def test_rejects_source_mismatch_before_executing_history(tmp_path):
    row,workspace=fixture(tmp_path);Image.new('RGB',(8,6),'white').save(workspace/'source.png')
    try:replay_prefix(row,workspace)
    except ValueError as e:assert 'source' in str(e)
    else:raise AssertionError('Mismatched source accepted')
    assert not(workspace/'region.json').exists()
