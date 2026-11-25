"""Execute saved next-action predictions under their verified teacher history.

This isolates localization and RGB computation errors. It is teacher-conditioned
diagnostic evidence, never an autonomous rollout or a semantic success metric.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import execute_action, initial_messages, parse_call, validate_outputs
from pixel.image_observations import load_observation


def pixel_comparison(source, reference, prediction):
    if any(a.dtype != np.uint8 or a.shape != source.shape for a in (reference, prediction)):
        raise ValueError('Comparisons require matching native uint8 RGB arrays.')
    expected_change = np.any(reference != source, axis=-1)
    actual_change = np.any(prediction != source, axis=-1)
    difference = np.abs(prediction.astype(np.int16) - reference.astype(np.int16))
    union = int((expected_change | actual_change).sum())
    outside = actual_change & ~expected_change
    return {
        'exact_reference_rgb': bool(np.array_equal(reference, prediction)),
        'reference_changed_pixels': int(expected_change.sum()),
        'predicted_changed_pixels': int(actual_change.sum()),
        'changed_footprint_iou': float((expected_change & actual_change).sum() / union) if union else None,
        'changed_footprint_recall': float((expected_change & actual_change).sum() / expected_change.sum()) if expected_change.any() else None,
        'changed_outside_reference_footprint': int(outside.sum()),
        'rgb_mae': float(difference.mean()),
        'rgb_max_absolute_error': int(difference.max()),
        'rgb_mae_on_reference_changes': float(difference[expected_change].mean()) if expected_change.any() else None,
        'footprint_definition': 'any actual uint8 RGB change; diagnostic, not object-mask or semantic accuracy',
    }


def box_comparison(reference, prediction):
    x0, y0 = max(reference[0], prediction[0]), max(reference[1], prediction[1])
    x1, y1 = min(reference[2], prediction[2]), min(reference[3], prediction[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    reference_area = (reference[2] - reference[0]) * (reference[3] - reference[1])
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    return {'reference_source_box': reference, 'predicted_source_box': prediction,
            'box_iou': intersection / (reference_area + prediction_area - intersection),
            'reference_box_coverage': intersection / reference_area}


def function_of(message):
    calls = message.get('tool_calls')
    if calls:
        if len(calls) != 1:
            raise ValueError('Historical assistant must contain one action.')
        f = calls[0]['function']
        return f['name'], f['arguments']
    return parse_call(message['content'])


def initialize_history(row, folder):
    folder.mkdir(parents=True, exist_ok=True)
    source_block = next(b for m in row['messages'] if m['role'] == 'user'
                        for b in m.get('content', []) if isinstance(b, dict) and b.get('type') == 'image')
    image = load_observation(source_block)
    image.save(folder / 'source.png')
    initial_messages(folder / 'source.png', 'diagnostic replay')
    source = np.asarray(image)
    step = 0
    for message in row['messages']:
        if message['role'] != 'assistant':
            continue
        name, arguments = function_of(message)
        feedback, passed = execute_action(name, arguments, folder, step, True, True, True)
        if not passed:
            raise ValueError('Teacher history execution failed: ' + json.dumps(feedback))
        step += 1
    return source, step


def run_action(name, arguments, folder, step):
    feedback, passed = execute_action(name, arguments, folder, step, True, True, True)
    if not passed:
        return {'execution_ok': False, 'feedback': feedback}
    if name == 'python':
        try:
            receipt = validate_outputs(folder)
        except Exception as error:
            return {'execution_ok': True, 'valid_native_rgb': False, 'error': str(error)}
        return {'execution_ok': True, 'valid_native_rgb': True, 'receipt': receipt}
    if name == 'view_image' and arguments.get('bbox_2d') is not None:
        return {'execution_ok': True, 'source_box': json.loads((folder / 'region.json').read_text())['source_bbox_pixels']}
    return {'execution_ok': True}


def audit(rows, predictions, workspace):
    indexed = {row['id']: row for row in rows}
    cases = []
    for prediction in predictions:
        row = indexed[prediction['id']]
        expected_name, expected_arguments = function_of(row['target_message'])
        case = {'id': row['id'], 'expected': expected_name,
                'teacher_conditioned': True, 'exact_arguments': prediction.get('arguments_match', False)}
        try:
            if not prediction.get('sampled_eos') or not prediction.get('complete_call'):
                raise ValueError('No complete sampled action; refusing to execute a partial call.')
            name, arguments = parse_call(prediction['text'])
            case['predicted'] = name
            if name != expected_name:
                raise ValueError('Predicted action differs from expected action.')
            key = hashlib.sha256((row['id'] + prediction['text']).encode()).hexdigest()
            base = workspace / key
            if base.exists():
                raise ValueError('Diagnostic workspace must be fresh; choose a new --workspace.')
            source, step = initialize_history(row, base / 'reference')
            other_source, other_step = initialize_history(row, base / 'prediction')
            if not np.array_equal(source, other_source) or step != other_step:
                raise ValueError('Reference and prediction history differ.')
            reference = run_action(expected_name, expected_arguments, base / 'reference', step)
            actual = run_action(name, arguments, base / 'prediction', step)
            case.update(reference=reference, prediction=actual)
            if expected_name == 'python' and reference.get('valid_native_rgb') and actual.get('valid_native_rgb'):
                case['pixels'] = pixel_comparison(source,
                    np.load(base / 'reference/edited_rgb.npy', allow_pickle=False),
                    np.load(base / 'prediction/edited_rgb.npy', allow_pickle=False))
            elif expected_name == 'view_image' and reference.get('source_box') and actual.get('source_box'):
                case['box'] = box_comparison(reference['source_box'], actual['source_box'])
        except Exception as error:
            case['diagnostic_error'] = type(error).__name__ + ': ' + str(error)
        cases.append(case)
    counts = Counter()
    for case in cases:
        counts['observed'] += 1
        counts[case['expected'] + '_observed'] += 1
        if 'diagnostic_error' in case:
            counts['diagnostic_error'] += 1
        if case.get('prediction', {}).get('valid_native_rgb'):
            counts['valid_native_rgb'] += 1
        if case.get('pixels', {}).get('exact_reference_rgb'):
            counts['exact_reference_rgb'] += 1
        if case.get('box', {}).get('box_iou') == 1.0:
            counts['exact_source_box'] += 1
    return {'scope': 'teacher-conditioned next-action pixel/localization diagnostics',
            'not_autonomous_or_semantic_success': True, 'counts': dict(counts), 'cases': cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.data.read_text().splitlines() if s.strip()]
    predictions = json.loads(args.predictions.read_text())
    report = audit(rows, predictions, args.workspace)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
