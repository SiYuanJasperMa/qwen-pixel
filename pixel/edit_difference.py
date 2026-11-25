"""Measured source/result differences as evidence for the visual evaluator."""
import base64
import io

import numpy as np
from PIL import Image


def difference_evidence(source,edited,box=None):
    with Image.open(source)as image:a=np.asarray(image.convert('RGB')).astype(np.float32)
    with Image.open(edited)as image:b=np.asarray(image.convert('RGB')).astype(np.float32)
    if a.shape!=b.shape:raise ValueError('Difference evidence requires aligned native RGB images.')
    height,width=a.shape[:2]
    x0,y0,x1,y1=box if box is not None else (0,0,width,height)
    x0,y0,x1,y1=map(int,(x0,y0,x1,y1))
    if not(0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('Invalid evaluator crop box.')
    delta=b-a;absolute=np.abs(delta);changed=np.any(absolute>0,axis=2)
    target=absolute[y0:y1,x0:x1];target_changed=changed[y0:y1,x0:x1]
    outside=np.ones((height,width),dtype=bool);outside[y0:y1,x0:x1]=False
    stats={'scope':'annotated_instance_box'if box is not None else'full_image',
        'changed_pixel_fraction':float(target_changed.mean()),
        'mean_absolute_rgb_delta':float(target.mean()),
        'mean_signed_rgb_delta':float(delta[y0:y1,x0:x1].mean()),
        'mean_delta_among_changed_pixels':float(target[target_changed].mean())if target_changed.any()else 0.,
        'changed_pixels_outside_box':int(changed[outside].sum())if box is not None else None}
    # Fixed amplification, never rescale a small difference into a large one.
    image=Image.fromarray(np.clip(absolute[y0:y1,x0:x1]*4,0,255).astype(np.uint8))
    stream=io.BytesIO();image.save(stream,format='PNG')
    block={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}
    return stats,block
