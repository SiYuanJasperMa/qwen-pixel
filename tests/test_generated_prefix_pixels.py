import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from pixel.agent_protocol import assistant_call


def diagnostic():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_generated_prefix_pixels.py'
    spec = importlib.util.spec_from_file_location('prefix_pixels', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uint8_difference_and_overedit_are_not_semantic_success():
    module = diagnostic()
    source = np.full((4, 4, 3), 250, dtype=np.uint8)
    reference = source.copy()
    reference[1:3, 1:3] = 10
    no_op = module.pixel_comparison(source, reference, source)
    assert no_op['rgb_max_absolute_error'] == 240
    assert no_op['predicted_changed_pixels'] == 0
    assert no_op['changed_footprint_recall'] == 0
    overedit = module.pixel_comparison(source, reference, np.full_like(source, 10))
    assert overedit['changed_footprint_iou'] == .25
    assert overedit['changed_outside_reference_footprint'] == 12
    assert 'semantic_pass' not in overedit
    with pytest.raises(ValueError):
        module.pixel_comparison(source, reference, source.astype(np.float32))


def test_real_code_execution_and_partial_call_rejection(tmp_path):
    module = diagnostic()
    source = tmp_path / 'source.png'
    Image.fromarray(np.full((4, 4, 3), 50, dtype=np.uint8)).save(source)
    code = 'output = pixels.copy()\noutput[1:3,1:3] = 200\nemit_rgb(output)'
    row = {'id': 'code', 'messages': [{'role': 'user', 'content': [
        {'type': 'image', 'image': str(source)}]}],
        'target_message': assistant_call('python', {'code': code})}
    # The public helper returns structured calls, so serialize with native Qwen
    # syntax here; the runtime parser must accept a complete sampled call.
    text = '<tool_call>\n<function=python>\n<parameter=code>\n' + code + '\n</parameter>\n</function>\n</tool_call><|im_end|>'
    prediction = {'id': 'code', 'text': text, 'sampled_eos': True, 'complete_call': True,
                  'arguments_match': True}
    report = module.audit([row], [prediction], tmp_path / 'exact')
    assert report['counts']['valid_native_rgb'] == 1
    assert report['counts']['exact_reference_rgb'] == 1
    assert report['not_autonomous_or_semantic_success'] is True
    partial = {**prediction, 'sampled_eos': False}
    report = module.audit([row], [partial], tmp_path / 'partial')
    assert report['counts']['diagnostic_error'] == 1
    assert not (tmp_path / 'partial').exists()


def test_native_box_coverage_separates_oversized_box():
    result = diagnostic().box_comparison([1, 1, 3, 3], [0, 0, 4, 4])
    assert result['reference_box_coverage'] == 1
    assert result['box_iou'] == .25
