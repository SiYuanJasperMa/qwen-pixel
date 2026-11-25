"""Read the exact bytes of a historical image observation and verify its identity."""
import hashlib
import io
import re
from pathlib import Path

from PIL import Image


def load_observation(block):
    path=Path(block['image'])
    raw=path.read_bytes()
    match=re.fullmatch(r'(?:input_|view_\d+_)([0-9a-f]{64})',path.stem) if path.parent.name=='observations' else None
    named_hash=match.group(1)if match else None
    expected=block.get('sha256')or named_hash
    if named_hash and block.get('sha256')and named_hash!=block['sha256']:
        raise ValueError('Image observation metadata disagrees with snapshot identity: '+str(path))
    if expected and hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Historical image observation changed: '+str(path))
    # Decode these verified bytes, rather than opening the mutable path again.
    with Image.open(io.BytesIO(raw))as image:return image.convert('RGB')
