import re
import pytest

from train.lora_scope import target_pattern


def test_attention_scope_reaches_recurrent_context_projections_but_preserves_gates():
    pattern=target_pattern('attention')
    for module in ('q_proj','k_proj','v_proj','o_proj','in_proj_qkv','out_proj'):
        assert re.fullmatch(pattern,'model.language_model.layers.0.attn.'+module)
    for module in ('in_proj_a','in_proj_b','in_proj_z','gate_proj','up_proj','down_proj'):
        assert not re.fullmatch(pattern,'model.language_model.layers.0.attn.'+module)
    assert not re.fullmatch(pattern,'model.visual.blocks.0.attn.q_proj')


def test_existing_standard_scope_is_unchanged():
    assert target_pattern('standard')==r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):target_pattern('guessed')
