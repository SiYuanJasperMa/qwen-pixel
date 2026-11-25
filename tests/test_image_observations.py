import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel.image_observations import load_observation
from pixel.agent_protocol import initial_messages,execute_action


def test_overwritten_snapshot_is_rejected_at_training_read(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(12,8),(100,40,30)).save(source)
    messages=initial_messages(source,'Change the color.')
    block=messages[1]['content'][0]
    original=np.asarray(load_observation(block))
    assert np.all(original==[100,40,30])
    Image.new('RGB',(12,8),(50,60,70)).save(block['image'])
    with pytest.raises(ValueError,match='Historical image observation changed'):
        load_observation(block)


def test_legacy_snapshot_filename_is_checked_without_new_metadata(tmp_path):
    folder=tmp_path/'observations';folder.mkdir()
    source=tmp_path/'image.png';Image.new('RGB',(10,7),'red').save(source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot=folder/f'view_03_{digest}.png';snapshot.write_bytes(source.read_bytes())
    assert load_observation({'image':str(snapshot)}).size==(10,7)
    Image.new('RGB',(10,7),'blue').save(snapshot)
    with pytest.raises(ValueError,match='Historical image observation changed'):
        load_observation({'image':str(snapshot)})


def test_snapshot_metadata_and_name_must_agree(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(9,6),'red').save(source)
    block=initial_messages(source,'Change the color.')[1]['content'][0]
    block['sha256']='0'*64
    with pytest.raises(ValueError,match='metadata disagrees'):
        load_observation(block)


def test_runtime_view_metadata_reads_same_pixels_after_live_file_changes(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(9,6),'red').save(source)
    response,passed=execute_action('view_image',{'path':'source.png'},tmp_path,0)
    assert passed
    block=response['content'][1]
    Image.new('RGB',(9,6),'blue').save(source)
    assert np.all(np.asarray(load_observation(block))==[255,0,0])
