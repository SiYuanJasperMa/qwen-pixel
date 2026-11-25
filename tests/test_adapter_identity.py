import json

import pytest

from pixel.adapter_identity import validate_adapter_base


def test_different_base_with_same_directory_name_is_rejected(tmp_path):
    a=tmp_path/'old/Qwen';b=tmp_path/'new/Qwen';a.mkdir(parents=True);b.mkdir(parents=True)
    adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':str(a)}))
    assert validate_adapter_base(adapter,str(a))['verified']
    with pytest.raises(ValueError,match='base model mismatch'):
        validate_adapter_base(adapter,str(b))


def test_symlink_to_same_base_is_allowed(tmp_path):
    model=tmp_path/'model';model.mkdir();alias=tmp_path/'alias';alias.symlink_to(model,target_is_directory=True)
    adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':str(model)}))
    assert validate_adapter_base(adapter,str(alias))['verified']


def test_saved_training_model_must_agree_with_adapter_configuration(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'Qwen/VersionA'}))
    (adapter/'run_state.json').write_text(json.dumps({'arguments':{'model':'Qwen/VersionB'}}))
    with pytest.raises(ValueError,match='training metadata disagrees'):
        validate_adapter_base(adapter,'Qwen/VersionA')
