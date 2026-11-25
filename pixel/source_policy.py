"""Persistent source opt-outs apply to new work without deleting archives."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISABLED = {'instructpix2pix'}
_cached_policy = None


def disabled_sources():
    global _cached_policy
    path = ROOT/'data/editing_source_policy.json'
    now = time.monotonic()
    if _cached_policy is not None and _cached_policy[0] == path and now-_cached_policy[1] < 1:
        return set(_cached_policy[2])
    configured = json.loads(path.read_text()).get('disabled_sources', {}) if path.exists() else {}
    disabled = DEFAULT_DISABLED | set(configured)
    _cached_policy = (path,now,disabled)
    return set(disabled)


def dataset_of(row):
    meta = row.get('meta') or {}
    dataset = row.get('dataset') or meta.get('dataset')
    identity = row.get('task_id') or meta.get('task_id') or row.get('id','')
    if isinstance(dataset,str) and dataset.lower().replace('_','').replace('-','')=='instructpix2pix':
        return 'instructpix2pix'
    if isinstance(identity,str) and identity.startswith('instructpix2pix_'):
        return 'instructpix2pix'
    if (row.get('origin')or{}).get('repo','').startswith('timbrooks/instructpix2pix'):
        return 'instructpix2pix'
    return dataset


def source_disabled(row):
    return dataset_of(row) in disabled_sources()
