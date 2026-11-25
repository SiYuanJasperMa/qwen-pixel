import numpy as np
from PIL import Image,ImageDraw
import pytest

from pixel.integer_edit_protocol import integer_polygons,program,SYSTEM


def test_native_raster_is_exact_despite_shorter_integer_representation():
    box=[111,49,271,339];polygons=[[[.4988,.2926],[.5067,.1948],[.4713,.1708],[.5461,.1621],[.9558,.0817],[.4476,.9728]]]
    integers=integer_polygons(polygons,box)
    a=Image.new('L',(640,427));b=Image.new('L',a.size)
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    ImageDraw.Draw(a).polygon([(x0+u*w,y0+v*h)for u,v in polygons[0]],fill=255)
    ImageDraw.Draw(b).polygon([(x0+round(u*w/1000),y0+round(v*h/1000))for u,v in integers[0]],fill=255)
    assert np.array_equal(np.asarray(a),np.asarray(b))
    assert all(isinstance(v,int)for p in integers for point in p for v in point)
    assert 'integer (u,v)'in SYSTEM
    assert 'round(u*region_width/1000.0)'in program(integers,'brighten')


def test_unsupported_exact_geometry_is_rejected_instead_of_silently_resized():
    with pytest.raises(ValueError):integer_polygons([[[0,0],[1,0],[1,1]]],[0,0,1400,500])
