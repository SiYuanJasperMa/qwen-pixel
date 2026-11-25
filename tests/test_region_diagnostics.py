import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/analyze_agent_edit_regions.py'
    spec=importlib.util.spec_from_file_location('region_diagnostics',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_support_is_separate_from_opacity_and_detects_spill():
    reference=np.zeros((4,4),dtype=bool);reference[1:3,1:3]=True
    alpha=reference.astype(np.float32);alpha[0,0]=.5
    a=module().region_metrics(alpha,reference)
    b=module().region_metrics(alpha*.25,reference)
    assert a['support_iou']==.8 and a['target_support_coverage']==1
    assert a['outside_target_support_pixels']==1
    assert a['support_iou']==b['support_iou']
    assert b['raw_alpha_peak']==.25 and b['raw_alpha_mean_on_target']==.25


def test_missing_target_and_invalid_alpha_are_distinct():
    reference=np.ones((2,2),dtype=bool);alpha=np.ones((2,2),dtype=np.float32);alpha[0,0]=0
    result=module().region_metrics(alpha,reference)
    assert result['target_support_coverage']==.75 and result['support_iou']==.75
    with pytest.raises(ValueError):module().region_metrics(np.full((2,2),2.),reference)
    with pytest.raises(ValueError):module().region_metrics(np.full((2,2),np.nan),reference)


def test_only_alpha_used_in_final_output_is_inspected():
    check=module().alpha_is_used_at_final_emit
    assert check('alpha_np = mask / 255\noutput = pixels * alpha_np\noutput = np.clip(output, 0, 255)\nemit_rgb(output)')
    assert not check('alpha_np = mask / 255\noutput = pixels * alpha_np\noutput = pixels.copy()\nemit_rgb(output)')
    assert not check('alpha_np = mask / 255\nemit_rgb(pixels)\nalpha_np = 0')
