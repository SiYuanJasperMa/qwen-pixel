"""Measured RGB preservation against evaluation-only annotated instances."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageFilter


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registry(tasks_file):
    path=Path(tasks_file).with_name('development_masks_eval_only.json')
    if not path.exists():return None
    registry=json.loads(path.read_text())
    if registry.get('tasks_sha256')!=file_digest(tasks_file):
        raise ValueError('Evaluation mask registry does not match the exact task file.')
    if registry.get('evaluation_only')is not True:
        raise ValueError('Instance masks must be marked evaluation-only.')
    return registry


def preservation_metrics(source,edited,mask,boundary_pixels=2):
    with Image.open(source)as image:a=np.asarray(image.convert('RGB')).astype(np.int16)
    with Image.open(edited)as image:b=np.asarray(image.convert('RGB')).astype(np.int16)
    with Image.open(mask)as image:reference=np.asarray(image.convert('L'))>0
    if a.shape!=b.shape or reference.shape!=a.shape[:2]:
        raise ValueError('Source, edited RGB and instance mask must have the same native geometry.')
    if not reference.any():raise ValueError('Evaluation instance mask is empty.')
    expanded=Image.fromarray(reference.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(2*boundary_pixels+1))
    allowed=np.asarray(expanded)>0
    delta=np.abs(b-a);outside=~allowed;energy=float(delta.sum())
    return {'boundary_tolerance_pixels':boundary_pixels,
        'reference_pixels':int(reference.sum()),
        'outside_target_changed_pixels_gt1':int(np.any(delta>1,axis=2)[outside].sum()),
        'outside_target_max_rgb_delta':int(delta[outside].max())if outside.any()else 0,
        'outside_target_rgb_delta_energy_fraction':float(delta[outside].sum()/energy)if energy else None,
        'target_changed_pixel_fraction_gt1':float(np.any(delta>1,axis=2)[reference].mean()),
        'target_mean_absolute_rgb_delta':float(delta[reference].mean()),
        'scope':'annotated instance with boundary tolerance; diagnostic alongside VLM judgment, not semantic pass'}


def evaluate_region(task,source,edited,registry):
    if registry is None:return None
    entry=registry['instances'].get(task['id'])
    if entry is None:raise ValueError('Task has no exact instance mask in the matched registry.')
    if entry['source_sha256']!=file_digest(task['image']):
        raise ValueError('Evaluation source changed since its instance mask was registered.')
    if entry['target_bbox_pixels']!=task['target_bbox_pixels']:
        raise ValueError('Instance mask belongs to a different target box.')
    with Image.open(source)as image:
        source_rgb_hash=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    if entry['source_rgb_sha256']!=source_rgb_hash:
        raise ValueError('Submitted source RGB differs from the registered evaluation image.')
    path=Path(entry['mask'])
    if entry['mask_sha256']!=file_digest(path):raise ValueError('Evaluation instance mask changed.')
    result=preservation_metrics(source,edited,path)
    result['annotation_id']=entry['annotation_id']
    return result
