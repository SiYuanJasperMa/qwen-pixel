import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_semantic_repair_sft import clip_partial_polygons,raster


def test_clipped_contour_omits_target_without_painting_background():
    polygons=[[[.1,.1],[.9,.1],[.9,.9],[.65,.9],[.65,.45],[.35,.45],[.35,.9],[.1,.9]]]
    box=[0,0,100,100];whole=raster(polygons,box,(100,100))
    partial=raster(clip_partial_polygons(polygons,0,.3,True),box,(100,100))
    assert partial.any()and np.count_nonzero(partial)<np.count_nonzero(whole)
    assert not np.any(partial&~whole)
    assert np.count_nonzero(partial&whole)/np.count_nonzero(whole)<.35
