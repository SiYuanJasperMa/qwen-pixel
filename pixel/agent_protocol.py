"""Minimal pixel-editing actions and an executable local demo environment."""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = [
    {'type': 'function', 'function': {'name': 'python',
     'description': 'Execute Python in the task directory. Read source.png with PIL or NumPy; compute and save edited.png and edited_rgb.npy. Variables do not persist; files do. PIL, numpy, cv2, math and json are available.',
     'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}}},
    {'type': 'function', 'function': {'name': 'view_image',
     'description': 'Inspect a source, crop, or edited image saved in the task directory.',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'finish',
     'description': 'Submit the edited PNG and matching uint8 HxWx3 RGB .npy array.',
     'parameters': {'type': 'object', 'properties': {'image_path': {'type': 'string'}, 'rgb_path': {'type': 'string'}}, 'required': ['image_path', 'rgb_path']}}},
]

SYSTEM = '''You are Qwen Pixel, an agent that edits real photographs by computing RGB values in Python.
You receive the source photograph and an editing instruction. The same full-resolution image is available as source.png in your task directory.
Use python to read, inspect and change its pixels with PIL, NumPy, and cv2 (classical operations such as inpaint and seamlessClone are allowed). Use view_image to inspect crops or your result when useful.
Do not use image generation models, segmentation models, external services, or invented helper functions.
Preserve source dimensions and details outside the requested edit. Work with relative coordinates when needed.
Save the result as edited.png and the exact matching uint8 RGB array as edited_rgb.npy. Submit both with finish.
Keep calls concise. Python calls run in separate processes; use saved files to share state. Use only relative task-directory paths.'''

GROUNDED_GUIDANCE = '''
For a local edit, first identify the target in the SOURCE image. If predicting bbox_2d, use normalized 0-1000 coordinates and explicitly convert x to x*width/1000 and y to y*height/1000 before editing pixels. A crop has its own coordinate frame; map it back to the source.
Read RGB into a writable array with np.array(image).copy(). Cast to float32 before arithmetic, clip to 0-255, then return uint8.
Local transforms must use a target mask and preserve the full canvas. For PIL Image.composite, both images and the mask must have the same full source dimensions. A local blur must not blur the entire photograph.'''

RGB_SYSTEM = SYSTEM + GROUNDED_GUIDANCE + '''
Python already provides image (the original PIL image) and pixels (a writable uint8 HxWx3 copy of its original RGB values) on every call.
Compute your edited RGB array yourself, then call emit_rgb(array). emit_rgb only serializes your computed array to edited.png and edited_rgb.npy; it does not perform any edit.
For example, create a boolean target mask, compute a transformed array, and use np.where(mask[...,None], transformed, pixels). For blur you can compute np.array(image.filter(ImageFilter.GaussianBlur(radius))) and blend only inside your mask.
Use float arrays for arithmetic. emit_rgb rounds/clips RGB values to the 8-bit range and preserves native dimensions. Inspect the emitted result and finish when the requested edit is done.
Classical helpers are available in Python: load_image, save_image, crop, resize, recolor, composite, preserve, and inpaint(image, mask, radius=3). cv2 is importable for classical operations such as cv2.inpaint and cv2.seamlessClone.'''
RGB_TOOLS = copy.deepcopy(TOOLS)
RGB_TOOLS[0]['function']['description'] = 'Execute Python with image (original PIL image), pixels (writable original uint8 RGB array), np, PIL classes, cv2, and emit_rgb(array). You must compute the edited RGB values. emit_rgb only writes your computed array as PNG and .npy, without editing it. Separate calls reset globals; saved files persist.'
CHECKED_RGB_SYSTEM = RGB_SYSTEM + '''
In this checked environment, emit_rgb rejects out-of-range RGB values instead of silently clipping a broken computation. Compute in float32, normalize masks/alpha to 0-1 before blending, and explicitly clip intended color transforms with np.clip(out,0,255) before emit_rgb.
If a tool reports a numeric error, inspect the reported min/max values, fix the computation, and retry. Do not resubmit unchanged code. This validation only checks computed RGB values; it does not choose the region or perform an edit.'''
CHECKED_RGB_TOOLS = copy.deepcopy(RGB_TOOLS)
CHECKED_RGB_TOOLS[0]['function']['description'] = 'Execute Python with image, pixels, np, PIL classes, and emit_rgb(array). Compute the actual edited RGB yourself. emit_rgb rejects invalid dimensions, nonfinite numbers, or values outside 0-255; explicitly clip intentional transforms. Tool errors report numeric ranges. Variables reset between calls; files persist.'
NORMALIZED_RGB_SYSTEM = CHECKED_RGB_SYSTEM + '''
Specify object polygon and box coordinates in normalized 0-1000 image coordinates. Convert them to the full native pixel coordinate system before drawing a mask. Include your own Python conversion function in the generated program; no external region finder is available. Keep x scaled by image width and y by image height. Numeric ellipse centers/radii must be converted in the same way. This makes coordinate units explicit for different image dimensions.'''
ROI_SYSTEM = RGB_SYSTEM + '''
Before a local edit, inspect the target with view_image(path='source.png', bbox_2d=[x1,y1,x2,y2]). You choose the target box from the source image in normalized 0-1000 coordinates. The tool only reads the requested pixels; it does not find objects or edit them.
Check that the returned crop contains the requested object. Use the returned source_bbox_pixels to address the full original RGB array. If the crop is wrong, correct your box before editing. After computing and emitting the edit, inspect edited.png and finish.'''
ROI_TOOLS = copy.deepcopy(RGB_TOOLS)
ROI_TOOLS[1]['function']['parameters']['properties']['bbox_2d'] = {
    'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4,
    'description': 'Optional [x1,y1,x2,y2] normalized 0-1000 box. You predict the box; the tool reads that region.'}
ROI_TOOLS[1]['function']['description'] = 'View the requested file or a region you specify. With bbox_2d, return the unchanged source crop, exact source pixel coordinates, and RGB statistics. No object detection or image editing is performed.'
REGION_EDIT_SYSTEM = CHECKED_RGB_SYSTEM + '''
Before writing an edit program, locate the requested object in the source and call view_image with path=source.png and bbox_2d=[x1,y1,x2,y2] in normalized 0-1000 FULL-IMAGE coordinates.
The tool reads only the region you selected, returns its actual RGB statistics and crop, and saves its FULL-SOURCE pixel box in region.json. It does not find the object or edit anything.
Read region.json in Python. For mask polygons/boxes, use normalized 0-1000 coordinates WITHIN this selected region, then convert to full-source pixels using the region's offset and size. Include your own conversion function in the program. Numeric ellipse radii are distances and must not receive an origin offset.
Use the inspected object/crop and its pixel values to construct the edit. Preserve unrelated content, faces and skin when changing clothing color. Inspect edited.png and finish only after a real modification.'''
REGION_EDIT_TOOLS = copy.deepcopy(CHECKED_RGB_TOOLS)
REGION_EDIT_TOOLS[1] = copy.deepcopy(ROI_TOOLS[1])
REGION_EDIT_TOOLS[1]['function']['description'] += ' The actual selected region metadata is also saved to region.json for your Python program.'
REGION_EDIT_USER_GUIDANCE = '\nFirst locate the edit target in the source image. Call view_image with path=source.png and bbox_2d=[x1,y1,x2,y2] in normalized 0-1000 full-image coordinates. Then use the actual crop and region.json to compute the RGB edit in Python.'


def add_region_guidance(messages):
    for block in messages[1]['content']:
        if block.get('type')=='text':block['text']+=REGION_EDIT_USER_GUIDANCE
    return messages


def copy_source_image(source, destination):
    """Decode any source format and save lossless RGB PNG without resizing."""
    with Image.open(source) as im:
        im.convert('RGB').save(destination)


def initial_messages(source, instruction):
    source = Path(source).resolve()
    # Freeze the initial attachment, independently of the editable source file.
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    observations = source.parent / 'observations'
    observations.mkdir(exist_ok=True)
    frozen_source = observations / ('input_' + digest + source.suffix)
    if not frozen_source.exists():
        frozen_source.write_bytes(raw)
    (observations / 'input_reference.json').write_text(json.dumps(
        {'path': str(frozen_source), 'sha256': digest}))
    with Image.open(io.BytesIO(raw)) as im:
        width, height = im.size
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'edit_instruction':instruction, 'content': [
                {'type': 'image', 'image': str(frozen_source),'sha256':digest},
                {'type': 'text', 'text': f'Source file: source.png\nSource dimensions: width={width}, height={height}. All editing coordinates must refer to this full source. Map crop coordinates back to the source before editing.\nEDIT: ' + instruction}]}]


def assistant_call(name, arguments):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': name, 'arguments': arguments}}]}


def parse_call(text):
    # This environment executes one action per assistant turn. Never silently
    # select the first action from an incomplete or ambiguous generation.
    if text.count('<tool_call>') != 1 or text.count('</tool_call>') != 1:
        raise ValueError('Expected exactly one complete native Qwen tool call.')
    outer = re.search(r'<tool_call>(.*?)</tool_call>', text, re.S)
    match = re.fullmatch(r'\s*<function=([A-Za-z_]\w*)>(.*?)</function>\s*',
                         outer.group(1), re.S) if outer else None
    if match is None:
        raise ValueError('Expected a native Qwen <tool_call><function=...> call.')
    name = match[1]
    schema = {tool['function']['name']: tool['function']['parameters'] for tool in ROI_TOOLS}
    if name not in schema:
        raise ValueError('Unknown action: ' + name)
    arguments = {}
    body = match[2].strip()
    while body:
        parameter = re.match(r'<parameter=([A-Za-z_]\w*)>(.*?)</parameter>', body, re.S)
        if parameter is None:
            raise ValueError('Malformed or extra content in tool-call parameters.')
        key, value = parameter.groups()
        if key in arguments:
            raise ValueError('Duplicate tool-call parameter: ' + key)
        if key not in schema[name]['properties']:
            raise ValueError('Unknown tool-call parameter: ' + key)
        # Qwen adds one delimiter newline around argument values. Preserve
        # actual code indentation and leading/trailing blank lines in the value.
        if value.startswith('\r\n'):
            value = value[2:]
        elif value.startswith('\n'):
            value = value[1:]
        if value.endswith('\r\n'):
            value = value[:-2]
        elif value.endswith('\n'):
            value = value[:-1]
        arguments[key] = value
        body = body[parameter.end():].lstrip()
    missing = set(schema[name].get('required', [])) - arguments.keys()
    if missing:
        raise ValueError('Missing tool-call parameters: ' + ', '.join(sorted(missing)))
    if name == 'view_image' and 'bbox_2d' in arguments:
        # The native template serializes typed arrays as JSON, not as a string
        # argument. Keep training tool-call objects and runtime objects aligned.
        arguments['bbox_2d'] = json.loads(arguments['bbox_2d'])
    return name, arguments


def local_path(workspace, value):
    path = (workspace / value).resolve()
    if not path.is_relative_to(workspace.resolve()):
        raise ValueError('File must be inside the task directory.')
    return path


def validate_outputs(workspace, image_path='edited.png', rgb_path='edited_rgb.npy'):
    image = np.asarray(Image.open(local_path(workspace, image_path)).convert('RGB'))
    rgb = np.load(local_path(workspace, rgb_path), allow_pickle=False)
    source = np.asarray(Image.open(workspace / 'source.png').convert('RGB'))
    reference = workspace / 'observations/input_reference.json'
    if reference.exists():
        expected = json.loads(reference.read_text())
        if hashlib.sha256((workspace / 'source.png').read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('The original source file was modified; submit a separate edited image.')
    if rgb.dtype != np.uint8 or rgb.shape != source.shape:
        raise ValueError('Expected uint8 RGB at the original image size.')
    if not np.array_equal(image, rgb):
        raise ValueError('PNG and RGB array differ.')
    changed = int(np.any(rgb != source, axis=-1).sum())
    if not changed:
        raise ValueError('The submitted image has no changed pixels.')
    return {'valid': True, 'shape': list(rgb.shape), 'dtype': str(rgb.dtype),
            'changed_pixels': changed, 'png_matches_rgb': True,
            'image_path': image_path, 'rgb_path': rgb_path,
            'image_sha256':hashlib.sha256(local_path(workspace,image_path).read_bytes()).hexdigest(),
            'rgb_sha256':hashlib.sha256(local_path(workspace,rgb_path).read_bytes()).hexdigest()}


def execute_action(name, arguments, workspace, step, rgb_profile=False, roi_profile=False, checked_rgb=False):
    """Return actual tool feedback. This is a trusted local smoke environment."""
    workspace = Path(workspace).resolve()
    try:
        if name == 'python':
            code = arguments['code']
            tree = ast.parse(code)
            allowed = {'PIL', 'numpy', 'math', 'json', 'colorsys', 'cv2'}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]]
                    if any(root not in allowed for root in roots):
                        raise ValueError('Use only PIL, numpy, cv2, math, json, or colorsys in this demo.')
            script = workspace / f'action_{step:02d}.py'
            prelude = 'import numpy as np\nimport cv2\ncv2.setNumThreads(1)\nfrom PIL import Image, ImageDraw, ImageFilter, ImageEnhance\n'
            prelude += '''def load_image(path): return Image.open(path).convert("RGB")
def save_image(image, path): image.save(path)
def crop(image, box): return image.crop(box)
def resize(image, size): return image.resize(size, Image.Resampling.LANCZOS)
def recolor(image, mask, color):
    a=np.array(image).copy(); m=np.array(mask.convert("L"))>127; a[m]=color; return Image.fromarray(a)
def composite(source, edited, mask): return Image.composite(edited, source, mask.convert("L"))
def preserve(source, edited, mask): return composite(source, edited, mask)
def inpaint(image, mask, radius=3):
    arr=np.array(image.convert("RGB"))
    m=np.array(mask.convert("L")) if hasattr(mask,"convert") else np.asarray(mask)
    m=(m>127).astype(np.uint8)*255
    return Image.fromarray(cv2.inpaint(arr,m,int(radius),cv2.INPAINT_TELEA))
'''
            if rgb_profile:
                prelude += '''import json
image = Image.open('source.png').convert('RGB')
pixels = np.array(image, dtype=np.uint8).copy()
def emit_rgb(array):
    rgb = np.asarray(array)
    if rgb.shape != pixels.shape or not np.isfinite(rgb).all():
        raise ValueError('Emit a finite HxWx3 RGB array at the original dimensions.')
    rgb = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    np.save('edited_rgb.npy', rgb)
    Image.fromarray(rgb).save('edited.png')
    result = {'saved': ['edited.png', 'edited_rgb.npy'], 'shape': list(rgb.shape), 'dtype': 'uint8'}
    print(json.dumps(result))
    return result
'''
            if checked_rgb:
                prelude += '''
_serialize_rgb = emit_rgb
def emit_rgb(array):
    values = np.asarray(array)
    if values.shape != pixels.shape or not np.isfinite(values).all():
        raise ValueError('Emit finite RGB at the original HxWx3 shape.')
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < -0.01 or maximum > 255.01:
        raise ValueError(f'Invalid RGB range min={minimum:.3f}, max={maximum:.3f}; expected 0-255. Check float arithmetic and normalize alpha/masks to 0-1; explicitly clip intentional transforms before emit_rgb.')
    return _serialize_rgb(values)
'''
            script.write_text(prelude + code)
            result = subprocess.run([sys.executable, str(script)], cwd=workspace,
                                    capture_output=True, text=True, timeout=30,
                                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                         'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
            feedback = {'exit_code': result.returncode, 'stdout': result.stdout[-6000:],
                        'stderr': result.stderr[-2000:]}
            return {'role': 'tool', 'name': name, 'content': json.dumps(feedback)}, result.returncode == 0
        if name == 'view_image':
            path = local_path(workspace, arguments['path'])
            raw = path.read_bytes()
            region_info = None
            with Image.open(path) as im:
                size = im.size
                if roi_profile and arguments.get('bbox_2d') is not None:
                    bbox = arguments['bbox_2d']
                    if isinstance(bbox, str):
                        bbox = json.loads(bbox)
                    if len(bbox) != 4 or not all(np.isfinite(float(v)) and 0 <= float(v) <= 1000 for v in bbox):
                        raise ValueError('Provide four finite normalized coordinates from 0 to 1000.')
                    box = [int(round(float(bbox[i]) * size[i % 2] / 1000)) for i in range(4)]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError('Target box must have positive width and height.')
                    crop = im.convert('RGB').crop(tuple(box))
                    values = np.asarray(crop)
                    region_info = {'source_size': list(size), 'source_bbox_pixels': box,
                        'crop_shape': list(values.shape),
                        'rgb_mean': np.round(values.mean(axis=(0,1)), 2).tolist(),
                        'rgb_min': values.min(axis=(0,1)).tolist(), 'rgb_max': values.max(axis=(0,1)).tolist()}
                    region_info['metadata_file']='region.json'
                    (workspace/'region.json').write_text(json.dumps(region_info))
                    import io
                    buffer = io.BytesIO(); crop.save(buffer, format='PNG'); raw = buffer.getvalue()
                else:
                    im.verify()
            digest = hashlib.sha256(raw).hexdigest()
            observations = workspace / 'observations'
            observations.mkdir(exist_ok=True)
            snapshot = observations / f'view_{step:02d}_{digest}{".png" if region_info else path.suffix}'
            snapshot.write_bytes(raw)
            return {'role': 'tool', 'name': name, 'content': [
                {'type': 'text', 'text': (json.dumps(region_info) if region_info else
                    f'Image returned: {arguments["path"]}; width={size[0]}, height={size[1]}. Viewing a crop does not change source image coordinates.')},
                {'type': 'image', 'image': str(snapshot),'sha256':digest}]}, True
        if name == 'finish':
            feedback = validate_outputs(workspace, arguments['image_path'], arguments['rgb_path'])
            return {'role': 'tool', 'name': name, 'content': json.dumps(feedback)}, True
        raise ValueError('Unknown action: ' + name)
    except Exception as error:
        return {'role': 'tool', 'name': name, 'content': json.dumps(
            {'error': type(error).__name__ + ': ' + str(error)})}, False


def export_sft_steps(messages, output, trajectory_id, provenance, tools=None):
    rows = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        rows.append({'id': f'{trajectory_id}_step_{len(rows):02d}', 'task': 'agent_edit',
                     'messages': messages[:i], 'target_message': message, 'tools': tools or TOOLS,
                     'meta': {'trajectory_id': trajectory_id, 'provenance': provenance}})
    Path(output).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return len(rows)
