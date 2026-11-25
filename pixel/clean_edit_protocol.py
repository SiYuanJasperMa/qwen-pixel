"""Compact, type-consistent RGB-edit supervision; no hidden edit API."""
import copy
import json

from .agent_protocol import REGION_EDIT_TOOLS,initial_messages

TOOLS=copy.deepcopy(REGION_EDIT_TOOLS)
TOOLS[0]['function']['description']='Execute Python. image is the original PIL RGB image; pixels is its writable uint8 HxWx3 array. PIL, numpy, and json are available. emit_rgb(array) validates and saves your computed RGB as edited.png and edited_rgb.npy; it does not perform an edit. Files persist, variables reset.'

SYSTEM='''You edit real photographs by computing RGB values in Python. No image generator, learned renderer, segmentation service, or hidden object-finding function is available.
The source is attached and available as source.png, image (PIL), and pixels (uint8 NumPy RGB).
First locate the entire visible requested object and call view_image(source.png, bbox_2d), with bbox_2d in 0-1000 FULL-IMAGE coordinates. The tool reads only your selected region and saves its actual source pixel box in region.json.
Then generate a Python program. Mask polygon points are (u,v) fractions in 0-1 WITHIN the selected region; convert using the region's source offset and size. These fractions are different from the full-image bbox_2d units.
Keep PIL images and NumPy arrays in distinctly named variables. Use ImageDraw on a PIL mask, ImageFilter on PIL images, and float32 NumPy arrays for RGB arithmetic. Normalize mask alpha to 0-1 before blending.
Compute the full requested edit, preserve details and all unrelated pixels, explicitly clip intended RGB transforms to 0-255, and call emit_rgb with the resulting full-size RGB array.
Inspect edited.png and submit edited.png plus edited_rgb.npy with finish. Tool errors are actual execution feedback; fix the reported computation instead of repeating it. Python calls reset variables; use files to share state.'''

USER_GUIDANCE='\nLocate the entire visible target first. Read its actual region before editing. bbox_2d uses 0-1000 full-image coordinates; Python mask polygon points use 0-1 fractions of the selected region.'


def initial(source,instruction):
    messages=initial_messages(source,instruction);messages[0]['content']=SYSTEM
    for block in messages[1]['content']:
        if block.get('type')=='text':block['text']+=USER_GUIDANCE
    return messages


def program(polygons,operation,color='red'):
    """Return the entire program as supervision, never a runtime editing tool."""
    source='''import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

source_rgb_np = pixels.astype(np.float32)
height, width = pixels.shape[:2]
x0, y0, x1, y1 = json.load(open('region.json'))['source_bbox_pixels']
region_width, region_height = x1-x0, y1-y0
polygons_roi = '''+json.dumps(polygons)+'''
mask_pil = Image.new('L', (width, height), 0)
mask_draw = ImageDraw.Draw(mask_pil)
for polygon_roi in polygons_roi:
    polygon_source = [(x0+u*region_width, y0+v*region_height) for u,v in polygon_roi]
    mask_draw.polygon(polygon_source, fill=255)
soft_mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=0.7))
alpha_np = np.asarray(soft_mask_pil, dtype=np.float32)/255.0
inside_region_np = np.zeros((height, width), dtype=bool)
inside_region_np[y0:y1, x0:x1] = True
alpha_np = np.where(inside_region_np, alpha_np, 0.0)
'''
    if operation=='blur':
        source+='''blurred_pil = image.filter(ImageFilter.GaussianBlur(radius=6))
transformed_rgb_np = np.asarray(blurred_pil, dtype=np.float32)
'''
    elif operation=='brighten':
        source+='''transformed_rgb_np = 255.0*np.power(source_rgb_np/255.0, 0.75)
'''
    elif operation=='recolor':
        palette={'red':[1.,.25,.25],'green':[.25,1.,.25],
                 'blue':[.25,.25,1.],'purple':[.8,.25,1.]}
        source+='value_np = source_rgb_np.max(axis=2)\n'
        source+='tint_color_rgb_np = np.array('+repr(palette[color])+', dtype=np.float32)\n'
        source+='tinted_rgb_np = value_np[...,None]*tint_color_rgb_np\n'
        source+='transformed_rgb_np = source_rgb_np*0.75+tinted_rgb_np*0.25\n'
    else:raise ValueError('Unknown pixel operation: '+operation)
    return source+'''output_rgb_np = source_rgb_np*(1-alpha_np[...,None])+transformed_rgb_np*alpha_np[...,None]
output_rgb_np = np.clip(output_rgb_np, 0, 255)
rounded_rgb_np = np.rint(output_rgb_np).astype(np.uint8)
assert np.array_equal(rounded_rgb_np[~inside_region_np], pixels[~inside_region_np]), 'Unexpected edit outside the selected region'
emit_rgb(rounded_rgb_np)
'''
