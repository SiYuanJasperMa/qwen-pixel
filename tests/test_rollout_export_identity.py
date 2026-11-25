import copy
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_agent_demo import rollout_export_identity, export_rollout_sft
from pixel.agent_protocol import export_sft_steps


def test_distinct_rollout_exports_do_not_overwrite_id_indexed_training_rows(tmp_path):
    first = [{'role': 'system', 'content': 'Edit pixels.'},
             {'role': 'user', 'content': 'Make this car blue.'},
             {'role': 'assistant', 'content': 'first source-specific program'}]
    second = copy.deepcopy(first)
    second[-1]['content'] = 'different source-specific program'
    original = copy.deepcopy(first)
    ids = []
    for i, messages in enumerate([first, second]):
        identity = rollout_export_identity(messages, 'Qwen3.8-27B', 'frozen_adapter')
        path = tmp_path/f'{i}.jsonl'
        export_sft_steps(messages, path, identity, 'actual model rollout')
        row = json.loads(path.read_text())
        assert row['messages'] == messages[:-1]
        assert row['target_message'] == messages[-1]
        ids.append(row['id'])
    assert len(set(ids)) == 2
    assert first == original
    assert rollout_export_identity(first, 'Qwen3.8-27B', 'frozen_adapter') == \
        rollout_export_identity(copy.deepcopy(first), 'Qwen3.8-27B', 'frozen_adapter')


def test_saved_rollout_preserves_exact_sampled_tokens_and_masks_no_future_history(tmp_path):
    messages = [{'role': 'system', 'content': 'Edit pixels.'},
                {'role': 'user', 'content': 'Edit the source.'},
                {'role': 'assistant', 'content': 'a sampled action'}]
    samples = [{'messages': copy.deepcopy(messages[:2]), 'target_text': messages[-1]['content'],
                'target_token_ids': [19, 42, 99], 'tool_image_mode': 'tool'}]
    path = tmp_path/'trace.jsonl'
    export_rollout_sft(messages, samples, path, 'Qwen', 'adapter', [], 99)
    row = json.loads(path.read_text())
    assert row['target_token_ids'] == samples[0]['target_token_ids']
    assert row['messages'] == messages[:2]
    assert row['target_message'] == messages[-1]
    samples[0]['target_token_ids'] = [19, 42]
    invalid = tmp_path/'incomplete.jsonl'
    with pytest.raises(ValueError, match='complete sampled'):
        export_rollout_sft(messages, samples, invalid, 'Qwen', 'adapter', [], 99)
    assert not invalid.exists()
    samples[0]['target_token_ids'] = [19, 42, 99]
    samples[0]['messages'].append(messages[-1])
    with pytest.raises(ValueError, match='visible history'):
        export_rollout_sft(messages, samples, invalid, 'Qwen', 'adapter', [], 99)
