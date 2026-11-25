import pytest

from pixel.decoding import DecodePolicyModel,decoding_implementation_identity
from pixel.eval_manifest import prepare_manifest


def test_cached_evaluation_rejects_changed_loaded_decoder(monkeypatch,tmp_path):
    original=decoding_implementation_identity()
    prepare_manifest(tmp_path,{'weights':'same','decoder':original})
    def changed(self,**kwargs):
        kwargs=dict(kwargs)
        kwargs['use_cache']=False
        return self.actor.generate(**kwargs)
    monkeypatch.setattr(DecodePolicyModel,'generate',changed)
    current=decoding_implementation_identity()
    assert current['loaded_policy_sha256']!=original['loaded_policy_sha256']
    assert current['loaded_cache_factory_sha256']==original['loaded_cache_factory_sha256']
    with pytest.raises(ValueError,match='different model'):
        prepare_manifest(tmp_path,{'weights':'same','decoder':current})
