import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train'))
from geometry_loss import geometry_indices


class GeometryLossTests(unittest.TestCase):
    def test_bbox_geometry_is_selected_without_weighting_function_syntax(self):
        pieces=['<tool_call>\n<function=view_image>\n<parameter=','bbox_2d>','\n[100,',' 200, 700,',' 900]\n','</parameter>','\n</function>']
        selected=geometry_indices(pieces)
        self.assertEqual(selected,[2,3,4])

    def test_mask_points_are_selected_but_rgb_constants_and_code_are_not(self):
        pieces=['polygons_roi',' = ','[[0.1, 0.2],',' [0.8, 0.9]]','\n','mask_pil = Image.new(...)\n','alpha_np = mask/255.0\n']
        self.assertEqual(geometry_indices(pieces),[2,3])

    def test_regular_non_geometry_actions_keep_the_standard_loss(self):
        self.assertEqual(geometry_indices(['<function=finish>','edited.png','</function>']),[])


if __name__=='__main__':unittest.main()
