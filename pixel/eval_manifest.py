"""Prevent evaluation results from silently crossing model/input configurations."""
import hashlib
import json
import marshal
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_digest(function):
    """Hash the loaded function, even if its source file changes during a run."""
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def prepare_manifest(directory,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'eval_manifest.json'
    expected=fingerprint(config)
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('fingerprint')!=expected:
            raise ValueError('Evaluation directory belongs to a different model, input, or decoding configuration. Use a new output directory.')
        return existing
    if any(directory.glob('*/evaluation.json')):
        raise ValueError('Legacy cached results have no verified configuration manifest. Use a new output directory.')
    result={'fingerprint':expected,'config':config,'version':'qwen-pixel-eval-manifest-v1'}
    path.write_text(json.dumps(result,indent=2))
    return result


def output_budgets(prompt_tokens,initial,max_context=8192,ceiling=6144):
    available=min(ceiling,max_context-prompt_tokens)
    if available<=0:return []
    current=min(initial,available)
    if current<=0:raise ValueError('Generation budget must be positive.')
    result=[current]
    while current<available:
        current=min(current*2,available)
        result.append(current)
    return result
