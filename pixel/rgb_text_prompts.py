"""Shared RGB-text editing prompts, without model or GPU dependencies."""
from pathlib import Path

EDIT_SYSTEM = '''You edit the source image by writing a complete Python function
def edit(image) returning a PIL RGB Image. At execution, image is exactly the
source described in the input. Use PIL, NumPy and cv2 only;
no network, external files, pretrained models or tool calls.
Return one JSON object with visual_evidence, target_object, target_region,
editing_intent, editing_plan, preservation_constraints, executable_code and requires_inpaint.
Ground observations in the source information provided; do not invent image contents.
Apply the instruction to the correct region and preserve unrelated content.
Use source pixels to preserve texture, shading and natural boundaries.
Use w, h = image.size and relative coordinates rather than a fixed resolution.'''


def make_instruction(task, mode='rgb_text'):
    text = f'EDIT: {task["instruction"]}\nSource dimensions: width={task["width"]}, height={task["height"]}.\n'
    text += 'Your code receives this source as a PIL Image and may inspect its pixels computationally. Preserve unrelated content.\n'
    if mode=='rgb_text':
        text += '\nNo image is attached. The complete source is supplied below as lossless numeric RGB text.\nFULL SOURCE RGB VALUES:\n'+Path(task['rgb_text']).read_text()
    elif mode=='vl':
        text += '\nThe attached image is the source.\n'
    else:
        text += '\nThis control receives no source pixel values before writing code. Do not claim to have observed the image.\n'
    return text


