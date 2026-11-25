"""Canonical task/source identities shared by all construction modes."""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical_task(sid):
    return sid.split('__')[0]


def source_group(sid, image):
    match = re.match(r'magicbrush_(?:train|dev|test)_(\d+)_', canonical_task(sid))
    if match:
        return 'magicbrush/image/' + match.group(1)
    p = Path(image)
    if not p.is_absolute():
        p = ROOT / p
    return 'source_png_sha256/' + hashlib.sha256(p.read_bytes()).hexdigest()


def split_of_group(group):
    return 'val' if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
