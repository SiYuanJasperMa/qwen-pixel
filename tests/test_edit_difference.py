import numpy as np
from PIL import Image
import pytest

from pixel.edit_difference import difference_evidence


def test_known_native_delta_and_preservation(tmp_path):
    a=np.full((5,7,3),80,dtype=np.uint8);b=a.copy();b[1:4,2:5]+=20
    source=tmp_path/'source.png';edited=tmp_path/'edited.png'
    Image.fromarray(a).save(source);Image.fromarray(b).save(edited)
    stats,block=difference_evidence(source,edited,[2,1,5,4])
    assert stats['changed_pixel_fraction']==1
    assert stats['mean_absolute_rgb_delta']==20
    assert stats['mean_signed_rgb_delta']==20
    assert stats['changed_pixels_outside_box']==0
    assert block['image_url']['url'].startswith('data:image/png;base64,')


def test_noop_and_changed_background(tmp_path):
    a=np.full((5,7,3),80,dtype=np.uint8);b=a.copy();b[0,0]+=30
    source=tmp_path/'source.png';edited=tmp_path/'edited.png'
    Image.fromarray(a).save(source);Image.fromarray(b).save(edited)
    stats,_=difference_evidence(source,edited,[2,1,5,4])
    assert stats['changed_pixel_fraction']==0
    assert stats['mean_delta_among_changed_pixels']==0
    assert stats['changed_pixels_outside_box']==1


def test_reject_misaligned_images(tmp_path):
    source=tmp_path/'source.png';edited=tmp_path/'edited.png'
    Image.new('RGB',(4,5)).save(source);Image.new('RGB',(5,4)).save(edited)
    with pytest.raises(ValueError,match='aligned'):
        difference_evidence(source,edited)
