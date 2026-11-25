import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pixel.clean_edit_protocol import program
from pixel.agent_protocol import execute_action,initial_messages,validate_outputs


class CleanEditProgramTests(unittest.TestCase):
    def test_all_rgb_transforms_preserve_pixels_outside_the_object_region(self):
        for operation,color in [('blur','red'),('brighten','red'),('recolor','red'),('recolor','green'),('recolor','blue'),('recolor','purple')]:
            with self.subTest(operation=operation,color=color),tempfile.TemporaryDirectory()as directory:
                workspace=Path(directory)
                y,x=np.mgrid[:32,:40]
                source=np.stack([30+x*3,50+y*3,90+(x+y)%10*4],axis=-1).astype(np.uint8)
                Image.fromarray(source).save(workspace/'source.png');initial_messages(workspace/'source.png','Edit the region.')
                (workspace/'region.json').write_text(json.dumps({'source_bbox_pixels':[10,8,30,24]}))
                code=program([[[0,0],[1,0],[1,1],[0,1]]],operation,color)
                response,passed=execute_action('python',{'code':code},workspace,0,True,True,True)
                self.assertTrue(passed,response)
                self.assertTrue(validate_outputs(workspace)['valid'])
                edited=np.load(workspace/'edited_rgb.npy')
                outside=np.ones((32,40),dtype=bool);outside[8:24,10:30]=False
                self.assertTrue(np.array_equal(edited[outside],source[outside]))
                if operation=='brighten':self.assertGreater(edited[10:22,12:28].mean(),source[10:22,12:28].mean())


if __name__=='__main__':unittest.main()
