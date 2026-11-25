import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train.agent_validation import preserve_training_state, trainable_digest


def test_validation_restores_rng_modes_cache_and_parameters_after_error():
    actor = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    actor.config = SimpleNamespace(use_cache=False)
    actor.train()
    actor[1].eval()
    actor.rope_deltas = torch.ones(1)
    before = trainable_digest(actor)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match='validation interrupted'):
        with preserve_training_state(actor):
            assert not actor.training and not actor[0].training
            random.random();np.random.random();torch.rand(4)
            actor.config.use_cache = True
            actor.rope_deltas = torch.zeros(1)
            raise RuntimeError('validation interrupted')
    assert actor.training and actor[0].training and not actor[1].training
    assert actor.config.use_cache is False and actor.rope_deltas is None
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert trainable_digest(actor) == before


def test_full_validation_wrapper_protects_lazy_setup_rng(monkeypatch):
    import train.agent_validation as validation
    actor = torch.nn.Linear(2,2)
    actor.config = SimpleNamespace(use_cache=False)
    actor.train()
    before = random.getstate()
    def setup(*_args):
        assert not actor.training
        random.random()
        raise RuntimeError('setup interrupted')
    monkeypatch.setattr(validation,'_evaluate',setup)
    with pytest.raises(RuntimeError,match='setup interrupted'):
        validation.evaluate(actor,None,None,None)
    assert actor.training and random.getstate() == before


def test_partial_or_completed_artifacts_cannot_be_used_as_fresh_rgb(tmp_path):
    from train.agent_validation import independent_workspace
    folder=tmp_path/'task'
    assert independent_workspace(folder)==folder
    folder.mkdir();(folder/'edited.png').write_bytes(b'old artifact')
    fresh=independent_workspace(folder)
    assert fresh.parent==folder and fresh!=folder and not fresh.exists()
    assert (folder/'edited.png').read_bytes()==b'old artifact'


def test_validator_accepts_decoder_module_without_new_convenience_export(monkeypatch):
    import pixel.decoding as decoding
    import pixel.region_preservation as region
    import train.agent_validation as validation
    monkeypatch.delattr(decoding,'decoding_implementation_identity')
    def stop_after_imports(*_):
        raise RuntimeError('past dependency imports')
    monkeypatch.setattr(region,'load_registry',stop_after_imports)
    actor=torch.nn.Linear(2,2);actor.config=SimpleNamespace(use_cache=False);actor.train()
    with pytest.raises(RuntimeError,match='past dependency imports'):
        validation.evaluate(actor,None,None,SimpleNamespace(agent_val_tasks='not_read'))
    assert actor.training and actor.config.use_cache is False
