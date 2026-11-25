"""Quality review for future curated snapshots; original judgments stay intact."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def review_registry():
    path = ROOT/'data/editing_quality_review.json'
    return json.loads(path.read_text()) if path.exists() else {'excluded_tasks':{}}


def task_excluded(row, review):
    meta = row.get('meta') or {}
    identity = meta.get('task_id') or row.get('task_id') or row.get('id')
    return identity in review.get('excluded_tasks', {})


def strong_image_grade(judge):
    def high(value):
        return isinstance(value, (int,float)) and not isinstance(value,bool) and 4 <= value <= 5
    return judge.get('pass') is True and high(judge.get('visual_quality')) and high(judge.get('preservation'))


def snapshot_quality_reason(row, review, record=None):
    from pixel.source_policy import source_disabled
    if source_disabled(row):
        return 'source_disabled_by_user'
    if task_excluded(row,review):
        return 'manual_quality_review'
    meta = row.get('meta') or {}
    dataset = meta.get('dataset') or ('instructpix2pix' if meta.get('task_id','').startswith('instructpix2pix') else '')
    if dataset != 'instructpix2pix':
        return None
    if record is None or record.get('pipeline_qualified') is not True:
        return 'instructpix2pix_missing_qualified_provenance'
    code = json.loads(row['answer']).get('executable_code','')
    if record.get('code_sha256') != hashlib.sha256(code.encode()).hexdigest():
        return 'instructpix2pix_quality_code_mismatch'
    if not strong_image_grade(record.get('judge') or {}):
        return 'instructpix2pix_visual_quality_or_preservation_below4'
    return None
