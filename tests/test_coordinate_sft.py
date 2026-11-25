import sys
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pixel.coordinate_sft import normalize_drawing_code


class CoordinateSFTTests(unittest.TestCase):
    def run_mask(self,code,width,height,frame=None):
        namespace={'pixels':np.zeros((height,width,3),dtype=np.uint8),
                   'Image':Image,'ImageDraw':ImageDraw,'np':np}
        if frame is not None:
            namespace['open']=lambda path:io.StringIO(json.dumps({'source_bbox_pixels':frame}))
        exec(code,namespace)
        return np.asarray(namespace['mask'])

    def test_normalized_draw_preserves_pixels_on_a_nonsquare_canvas(self):
        code='''height,width=pixels.shape[:2]
mask=Image.new('L',(width,height),0)
d=ImageDraw.Draw(mask)
body=[(11,7),(96,29),(55,113),(14,78)]
d.polygon(body,fill=200)
d.ellipse((12,17,64,82),fill=220)
d.rectangle((103,11,138,94),fill=110)
'''
        for width,height in [(573,319),(427,640)]:
            normalized,count=normalize_drawing_code(code,width,height)
            self.assertEqual(count,3)
            self.assertTrue(np.array_equal(self.run_mask(code,width,height),
                                           self.run_mask(normalized,width,height)))

    def test_numeric_ellipse_centers_and_radii_keep_the_same_mask(self):
        code='''height,width=pixels.shape[:2]
Y,X=np.mgrid[0:height,0:width]
cx,cy=124,71
rx,ry=35,18
mask=(((X-cx)/rx)**2+((Y-cy)/ry)**2<=1).astype(np.uint8)
'''
        normalized,count=normalize_drawing_code(code,427,640)
        self.assertEqual(count,2)
        self.assertTrue(np.array_equal(self.run_mask(code,427,640),
                                       self.run_mask(normalized,427,640)))

    def test_roi_offsets_and_ellipse_radii_are_distinct_coordinate_types(self):
        code='''height,width=pixels.shape[:2]
Y,X=np.mgrid[0:height,0:width]
cx,cy=124,71
rx,ry=35,18
mask=(((X-cx)/rx)**2+((Y-cy)/ry)**2<=1).astype(np.uint8)
'''
        frame=[50,20,180,130]
        normalized,count=normalize_drawing_code(code,427,640,frame)
        self.assertEqual(count,2)
        self.assertIn('_span',normalized)
        self.assertTrue(np.array_equal(self.run_mask(code,427,640),
                                       self.run_mask(normalized,427,640,frame)))


if __name__=='__main__':unittest.main()
