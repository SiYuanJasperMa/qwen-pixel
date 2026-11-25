"""Aligned RGB smoke metrics, independent of benchmark-specific mask assets."""
import numpy as np
from PIL import Image


def pixel_metrics(source, target, edited):
    arrays = []
    for path in (source, target, edited):
        with Image.open(path) as im:
            arrays.append(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.)
    s, t, p = arrays
    if s.shape != t.shape or s.shape != p.shape:
        raise ValueError('RGB comparison requires identical image shapes')
    l1 = float(np.abs(p - t).mean())
    noop = float(np.abs(s - t).mean())
    return {'metric_protocol': 'aligned_resized_rgb_smoke_v1', 'l1': l1,
            'no_op_l1': noop, 'gain': noop - l1,
            'mse': float(np.square(p - t).mean()),
            'changed_pixel_fraction': float(np.any(p != s, axis=-1).mean())}


def summarize(records, audit):
    modes = {}
    for mode in {r['mode'] for r in records}:
        subset = [r for r in records if r['mode'] == mode]
        modes[mode] = {'tasks': len(subset), 'parsed': sum(r['parsed'] for r in subset),
                      'executed': sum(r.get('execution', {}).get('status') == 'passed' for r in subset),
                      'judge_pass': sum(bool(r.get('judge', {}).get('pass')) for r in subset),
                      'judge_pending': sum(bool(r.get('judge_pending')) for r in subset),
                      'input_tokens': [r.get('input_tokens') for r in subset]}
    return {'model_audit': audit, 'modes': modes}
