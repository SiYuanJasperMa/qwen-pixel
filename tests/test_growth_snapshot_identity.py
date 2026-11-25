import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_editing_growth_snapshot import namespaced_candidate, freeze


def test_legacy_id_collision_preserves_both_independent_answers():
    row = {'id': 'task__ti_gpt61_growth1', 'meta': {'task_id': 'task'},
           'answer': json.dumps({'executable_code': 'def edit(image): return image'})}
    a = namespaced_candidate(row, 'rft_rescue/export.jsonl')
    b = namespaced_candidate(row, 'independent_ti/export.jsonl')
    assert a['id'] != b['id']
    assert a['meta']['original_candidate_id'] == b['meta']['original_candidate_id'] == row['id']
    assert a == namespaced_candidate(row, 'rft_rescue/export.jsonl')
    changed = {**row, 'answer': json.dumps({'executable_code': 'def edit(image): return image.copy()'})}
    assert namespaced_candidate(changed, 'rft_rescue/export.jsonl')['id'] != a['id']
    assert 'construction_export' not in row['meta']


def test_existing_frozen_training_payload_cannot_be_replaced(tmp_path):
    path = tmp_path/'train.jsonl'
    freeze(path, 'original\n')
    freeze(path, 'original\n')
    with pytest.raises(ValueError, match='overwrite frozen'):
        freeze(path, 'different teacher answer\n')
    assert path.read_text() == 'original\n'
