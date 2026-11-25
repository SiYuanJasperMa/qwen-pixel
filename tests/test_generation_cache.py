import torch
from transformers.cache_utils import LinearAttentionLayer, DynamicLayer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pixel.generation_cache import RecurrentPrecisionLayer, fresh_precision_cache


def test_installed_cache_rounding_and_local_precision_preservation():
    conv = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    state = torch.full((1, 2, 3, 4), 1.001, dtype=torch.float32)
    original, fixed = LinearAttentionLayer(), RecurrentPrecisionLayer()
    for layer in [original, fixed]:
        layer.update_conv_state(conv)
        layer.update_recurrent_state(state)
    assert original.recurrent_states.dtype == torch.bfloat16
    assert not torch.equal(original.recurrent_states.float(), state)
    assert fixed.conv_states.dtype == torch.bfloat16
    assert fixed.recurrent_states.dtype == torch.float32
    assert torch.equal(fixed.recurrent_states, state)
    updated = state + .0001
    address = fixed.recurrent_states.data_ptr()
    fixed.update_recurrent_state(updated)
    assert fixed.recurrent_states.data_ptr() == address
    assert torch.equal(fixed.recurrent_states, updated)
    fixed.reset()
    assert fixed.recurrent_states.dtype == torch.float32
    assert not fixed.recurrent_states.any()


def test_fresh_hybrid_cache_leaves_full_attention_layers_intact():
    config = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'])
    first, second = fresh_precision_cache(config), fresh_precision_cache(config)
    assert isinstance(first.layers[0], RecurrentPrecisionLayer)
    assert isinstance(first.layers[1], DynamicLayer)
    assert first.layers[0] is not second.layers[0]
    conv = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    state = torch.full((1, 2, 3, 4), 1.001, dtype=torch.float32)
    first.update_conv_state(conv, 0)
    first.update_recurrent_state(state, 0)
    assert first.has_previous_state(0)
    assert not second.has_previous_state(0)
    assert torch.equal(first.layers[0].recurrent_states, state)
