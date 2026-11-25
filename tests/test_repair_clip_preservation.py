import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_semantic_repair_sft import preserve_original_clip
from pixel.clean_edit_protocol import program
from pixel.roi_augmentation import recrop_polygons


def test_recrop_preserves_actual_rgb_including_feathered_boundary(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=np.full((12,12,3),100,dtype=np.uint8);image=Image.fromarray(source)
    original=[3,3,8,8];expanded=[1,1,10,10];polygon=[[[0,0],[1,0],[1,1],[0,1]]]
    def execute(code,box):
        Path('region.json').write_text(json.dumps({'source_bbox_pixels':box}))
        outputs=[];scope={'pixels':source.copy(),'image':image,'emit_rgb':lambda x:outputs.append(x.copy())}
        exec(code,scope)
        return outputs[0]
    expected=execute(program(polygon,'brighten'),original)
    mapped=recrop_polygons(polygon,original,expanded)
    uncorrected=execute(program(mapped,'brighten'),expanded)
    actual=execute(preserve_original_clip(program(mapped,'brighten'),original),expanded)
    assert not np.array_equal(expected,uncorrected)
    assert np.array_equal(expected,actual)


def test_unknown_program_clip_is_rejected():
    try:preserve_original_clip('emit_rgb(pixels)',[0,0,8,8])
    except ValueError:pass
    else:raise AssertionError('Unknown program modified silently')
