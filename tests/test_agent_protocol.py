import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel.agent_protocol import copy_source_image, execute_action, initial_messages, parse_call, validate_outputs
from pixel.reward import score_judgment


class AgentProtocolTests(unittest.TestCase):
    def test_incomplete_or_multiple_calls_are_not_silently_executed(self):
        call='<tool_call>\n<function=view_image>\n<parameter=path>\nsource.png\n</parameter>\n</function>\n</tool_call>'
        for text in [call.removesuffix('</tool_call>'),call+'\n'+call,
                     call.replace('</function>','</function>\n<function=finish></function>')]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):parse_call(text)

    def test_duplicate_unknown_and_missing_parameters_are_rejected(self):
        call='<tool_call>\n<function=view_image>\n<parameter=path>\nsource.png\n</parameter>\n</function>\n</tool_call>'
        for text in [call.replace('</function>','<parameter=path>\nedited.png\n</parameter>\n</function>'),
                     call.replace('parameter=path','parameter=filename'),
                     '<tool_call><function=view_image></function></tool_call>']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):parse_call(text)

    def test_native_python_argument_preserves_actual_blank_lines(self):
        code='\n\ndef f():\n    return 1\n\n'
        text='<tool_call>\n<function=python>\n<parameter=code>\n'+code+'\n</parameter>\n</function>\n</tool_call>'
        self.assertEqual(parse_call(text),('python',{'code':code}))

    def test_partial_face_edit_cannot_pass_a_full_object_rubric(self):
        result=score_judgment({'instruction_adherence':4,'preservation':5,'visual_quality':4,
            'pass':True,'rubric_version':'full_visible_target_v1',
            'target_extent_score':2,'target_extent_complete':False})
        self.assertFalse(result['pass'])
        self.assertEqual(result['reward'],0)
    def test_native_region_call_preserves_typed_array_arguments(self):
        text='<tool_call>\n<function=view_image>\n<parameter=path>\nsource.png\n</parameter>\n<parameter=bbox_2d>\n[100, 200, 600, 900]\n</parameter>\n</function>\n</tool_call>'
        name,arguments=parse_call(text)
        self.assertEqual(name,'view_image')
        self.assertEqual(arguments,{'path':'source.png','bbox_2d':[100,200,600,900]})
    def test_checked_rgb_reports_mask_scale_error_without_saving_a_corrupt_image(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            Image.fromarray(np.full((12,16,3),100,dtype=np.uint8)).save(workspace/'source.png')
            code='a=np.full(pixels.shape[:2],200,dtype=np.float32)\nout=pixels.astype(np.float32)*(1-a[...,None])+150*a[...,None]\nemit_rgb(out)'
            response, passed = execute_action('python',{'code':code},workspace,0,True,False,True)
            self.assertFalse(passed)
            self.assertIn('normalize alpha/masks',json.loads(response['content'])['stderr'])
            self.assertFalse((workspace/'edited.png').exists())
            fixed=code.replace('a=np.full(pixels.shape[:2],200,dtype=np.float32)',
                               'a=np.full(pixels.shape[:2],200,dtype=np.float32)/255.0')
            response, passed = execute_action('python',{'code':fixed},workspace,1,True,False,True)
            self.assertTrue(passed,response)
            self.assertTrue(validate_outputs(workspace)['valid'])
    def test_region_reader_returns_only_requested_source_pixels_and_native_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = np.arange(8*12*3, dtype=np.uint8).reshape(8,12,3)
            Image.fromarray(source).save(workspace / 'source.png')
            response, passed = execute_action('view_image', {'path':'source.png','bbox_2d':'[250,250,750,750]'}, workspace, 0, True, True)
            self.assertTrue(passed, response)
            information = json.loads(response['content'][0]['text'])
            self.assertEqual(information['source_bbox_pixels'], [3,2,9,6])
            self.assertTrue(np.array_equal(np.asarray(Image.open(response['content'][1]['image'])), source[2:6,3:9]))
            self.assertTrue(np.array_equal(np.asarray(Image.open(workspace / 'source.png')), source))
    def test_rgb_emitter_serializes_only_the_agent_computed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = np.full((12, 16, 3), 100, dtype=np.uint8)
            Image.fromarray(source).save(workspace / 'source.png')
            initial_messages(workspace / 'source.png', 'Change one region.')
            feedback, passed = execute_action('python', {'code': 'out = pixels.copy()\nout[2:4, 5:9] = [30, 90, 200]\nemit_rgb(out)'}, workspace, 0, True)
            self.assertTrue(passed, feedback)
            out = np.load(workspace / 'edited_rgb.npy')
            self.assertTrue(np.array_equal(out[2:4, 5:9], np.broadcast_to([30,90,200], (2,4,3))))
            self.assertEqual(validate_outputs(workspace)['changed_pixels'], 8)
    def test_native_jpeg_input_is_decoded_to_png_without_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            image = np.full((19, 31, 3), [100, 170, 30], dtype=np.uint8)
            Image.fromarray(image).save(workspace / 'original.jpg')
            copy_source_image(workspace / 'original.jpg', workspace / 'source.png')
            with Image.open(workspace / 'source.png') as im:
                self.assertEqual(im.size, (31, 19))
                self.assertEqual(im.format, 'PNG')
                self.assertTrue(np.array_equal(np.asarray(im), np.asarray(Image.open(workspace / 'original.jpg'))))

    def test_source_cannot_be_replaced_to_cheat_the_submission_check(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = np.full((12, 16, 3), 100, dtype=np.uint8)
            Image.fromarray(source).save(workspace / 'source.png')
            initial_messages(workspace / 'source.png', 'Change one region.')
            modified = source.copy(); modified[:5, :, 0] = 200
            Image.fromarray(modified).save(workspace / 'source.png')
            Image.fromarray(source).save(workspace / 'edited.png')
            np.save(workspace / 'edited_rgb.npy', source)
            with self.assertRaisesRegex(ValueError, 'source file was modified'):
                validate_outputs(workspace)
    def test_returned_image_is_immutable_when_live_result_is_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            before = np.full((8, 9, 3), 70, dtype=np.uint8)
            Image.fromarray(before).save(workspace / 'edited.png')
            response, passed = execute_action('view_image', {'path': 'edited.png'}, workspace, 1)
            self.assertTrue(passed)
            snapshot = Path(response['content'][1]['image'])
            Image.fromarray(np.full_like(before, 200)).save(workspace / 'edited.png')
            self.assertTrue(np.array_equal(np.asarray(Image.open(snapshot)), before))
    def test_wrong_target_gets_zero_reward_even_if_quality_has_nonzero_score(self):
        wrong = score_judgment({'instruction_adherence': 1, 'preservation': 1,
            'visual_quality': 2, 'pass': False})
        self.assertGreater(wrong['raw_reward'], 0)
        self.assertEqual(wrong['reward'], 0)
        self.assertFalse(wrong['pass'])
        good = score_judgment({'instruction_adherence': 4, 'preservation': 4,
            'visual_quality': 3, 'pass': True})
        self.assertAlmostEqual(good['reward'], .725)
        self.assertTrue(good['pass'])
    def test_multiline_native_python_call_preserves_indentation(self):
        text = '<tool_call>\n<function=python>\n<parameter=code>\ndef f():\n    return 1\nprint(f())\n</parameter>\n</function>\n</tool_call>'
        name, arguments = parse_call(text)
        self.assertEqual(name, 'python')
        self.assertEqual(arguments['code'], 'def f():\n    return 1\nprint(f())')

    def test_submission_rejects_noop_and_mismatched_pixel_file(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = np.full((12, 16, 3), 100, dtype=np.uint8)
            Image.fromarray(source).save(workspace / 'source.png')
            Image.fromarray(source).save(workspace / 'edited.png')
            np.save(workspace / 'edited_rgb.npy', source)
            with self.assertRaisesRegex(ValueError, 'no changed pixels'):
                validate_outputs(workspace)
            modified = source.copy(); modified[3:6, 2:8, 0] = 200
            np.save(workspace / 'edited_rgb.npy', modified)
            with self.assertRaisesRegex(ValueError, 'differ'):
                validate_outputs(workspace)
            Image.fromarray(modified).save(workspace / 'edited.png')
            self.assertEqual(validate_outputs(workspace)['changed_pixels'], 18)

    def test_failed_python_returns_stderr_and_view_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            response, passed = execute_action('python', {'code': 'raise ValueError("demo error")'}, workspace, 0)
            self.assertFalse(passed)
            self.assertIn('demo error', json.loads(response['content'])['stderr'])
            response, passed = execute_action('view_image', {'path': '../outside.png'}, workspace, 1)
            self.assertFalse(passed)
            self.assertIn('inside the task directory', json.loads(response['content'])['error'])


if __name__ == '__main__':
    unittest.main()
