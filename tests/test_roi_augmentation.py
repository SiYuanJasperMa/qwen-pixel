import numpy as np
from PIL import Image,ImageDraw
import pytest

from pixel.roi_augmentation import normalized_box,padded_box,recrop_polygons


def render(polygons,box):
    image=Image.new('L',(640,427));draw=ImageDraw.Draw(image);x0,y0,x1,y1=box
    for polygon in polygons:draw.polygon([(x0+u*(x1-x0),y0+v*(y1-y0))for u,v in polygon],fill=255)
    return np.asarray(image)


def test_changed_crop_preserves_actual_nonsquare_source_geometry():
    box=[100,80,300,380];polygons=[[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]]
    expanded=padded_box(box,(640,427),.2)
    mapped=recrop_polygons(polygons,box,expanded)
    # Bounds reflect the same real-image object, despite changed normalized values.
    a=render(polygons,box);b=render(mapped,expanded)
    assert np.array_equal(a,b)
    assert mapped!=polygons


def test_padding_clamps_to_native_edges_and_tool_coordinates_round_trip():
    box=[0,3,638,421];expanded=padded_box(box,(640,427),.1)
    assert expanded==[0,0,640,427]
    assert [round(v*(640,427)[i%2]/1000)for i,v in enumerate(normalized_box(expanded,(640,427)))]==expanded
    other=[23,73,227,422]
    assert [round(v*(640,427)[i%2]/1000)for i,v in enumerate(normalized_box(other,(640,427)))]==other


def test_invalid_padding_is_rejected():
    with pytest.raises(ValueError):padded_box([1,1,3,3],(10,10),-.1)
