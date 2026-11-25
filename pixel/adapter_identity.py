"""Reject adapters loaded on a different declared base model before GPU loading."""
import json
from pathlib import Path


def canonical_base(value):
    path=Path(value).expanduser()
    return str(path.resolve())if path.is_absolute()or path.exists()else str(value)


def validate_adapter_base(adapter,model):
    if not adapter:return None
    config=json.loads((Path(adapter)/'adapter_config.json').read_text())
    expected=config.get('base_model_name_or_path')
    if not expected:raise ValueError('Adapter does not declare its training base model.')
    actual=canonical_base(model)
    if canonical_base(expected)!=actual:
        raise ValueError(f'Adapter base model mismatch: trained on {expected}, requested {model}.')
    state=Path(adapter)/'run_state.json'
    if state.is_file():
        trained=json.loads(state.read_text()).get('arguments',{}).get('model')
        if trained and canonical_base(trained)!=actual:
            raise ValueError('Adapter training metadata disagrees with the requested base model.')
    return {'adapter':str(adapter),'declared_base':expected,'resolved_base':actual,'verified':True}
