import numpy as np
from PIL import Image
import hashlib
import pytest

from pixel.region_preservation import preservation_metrics,evaluate_region


def test_non_target_delta_uses_signed_rgb_and_ignores_boundary(tmp_path):
    source=np.full((12,12,3),250,dtype=np.uint8);edited=source.copy()
    mask=np.zeros((12,12),dtype=np.uint8);mask[4:8,4:8]=255
    edited[4:8,4:8]=10
    edited[3,4]=10  # one-pixel boundary spill is reported with tolerance
    edited[0,0]=10  # genuinely outside the allowed target boundary
    for name,array in [('source',source),('edited',edited),('mask',mask)]:Image.fromarray(array).save(tmp_path/(name+'.png'))
    result=preservation_metrics(tmp_path/'source.png',tmp_path/'edited.png',tmp_path/'mask.png')
    assert result['outside_target_changed_pixels_gt1']==1
    assert result['outside_target_max_rgb_delta']==240
    assert result['target_changed_pixel_fraction_gt1']==1
    assert abs(result['outside_target_rgb_delta_energy_fraction']-1/18)<1e-7


def test_noop_is_reported_without_fabricated_energy_ratio(tmp_path):
    Image.fromarray(np.zeros((5,5,3),dtype=np.uint8)).save(tmp_path/'source.png')
    mask=np.zeros((5,5),dtype=np.uint8);mask[2,2]=255;Image.fromarray(mask).save(tmp_path/'mask.png')
    result=preservation_metrics(tmp_path/'source.png',tmp_path/'source.png',tmp_path/'mask.png')
    assert result['outside_target_rgb_delta_energy_fraction']is None
    assert result['target_changed_pixel_fraction_gt1']==0


def test_registry_rejects_wrong_source_rgb_and_changed_masks(tmp_path):
    source=tmp_path/'source.png';wrong=tmp_path/'wrong.png';mask=tmp_path/'mask.png'
    Image.new('RGB',(5,5),'red').save(source);Image.new('RGB',(5,5),'blue').save(wrong)
    Image.new('L',(5,5),255).save(mask)
    task={'id':'test','image':str(source),'target_bbox_pixels':[0,0,5,5]}
    entry={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'source_rgb_sha256':hashlib.sha256(Image.open(source).convert('RGB').tobytes()).hexdigest(),
           'target_bbox_pixels':[0,0,5,5],'mask':str(mask),'mask_sha256':hashlib.sha256(mask.read_bytes()).hexdigest(),'annotation_id':1}
    registry={'instances':{'test':entry}}
    with pytest.raises(ValueError,match='source RGB differs'):
        evaluate_region(task,wrong,wrong,registry)
    Image.new('L',(5,5),0).save(mask)
    with pytest.raises(ValueError,match='instance mask changed'):
        evaluate_region(task,source,source,registry)
