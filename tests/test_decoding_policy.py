from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pixel.decoding import DecodePolicyModel, configure_decoding
from pixel.generation_cache import RecurrentPrecisionLayer


def test_each_action_gets_an_independent_cache_and_real_actor_result(monkeypatch):
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    calls = []
    result = torch.tensor([[4, 5]])
    actor = SimpleNamespace(config=Qwen3_5TextConfig(num_hidden_layers=2,
        layer_types=['linear_attention', 'full_attention']), device='cpu',
        generate=lambda **kwargs: (calls.append(kwargs), result)[1], eval=lambda: None)
    model = DecodePolicyModel(actor, 'fp32_state')
    assert model.eval() is model and model.device == 'cpu'
    assert model.generate(input_ids=torch.tensor([[4]]), use_cache=True) is result
    assert model.generate(input_ids=torch.tensor([[4]]), use_cache=True) is result
    assert isinstance(calls[0]['past_key_values'].layers[0], RecurrentPrecisionLayer)
    assert calls[0]['past_key_values'] is not calls[1]['past_key_values']
    with pytest.raises(ValueError, match='fresh recurrent cache'):
        model.generate(past_key_values=calls[0]['past_key_values'])


def test_legacy_does_not_replace_incoming_cache():
    cache = object()
    actor = SimpleNamespace(generate=lambda **kwargs: kwargs)
    result = DecodePolicyModel(actor, 'legacy').generate(past_key_values=cache)
    assert result['past_key_values'] is cache


def test_cuda_workspace_must_be_set_before_initialization(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='before loading CUDA'):
        configure_decoding('deterministic')


def test_policy_requires_actual_determinism(monkeypatch):
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: False)
    with pytest.raises(ValueError, match='not configured'):
        DecodePolicyModel(SimpleNamespace(), 'fp32_state').generate()
