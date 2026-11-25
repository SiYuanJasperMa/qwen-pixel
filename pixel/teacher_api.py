"""Teacher/judge client using existing workspace credentials without copying them."""
import ast
import base64
import json
import io
import os
import re
from pathlib import Path

from openai import AzureOpenAI
from PIL import Image
from .reward import score_judgment
from .edit_difference import difference_evidence

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-5.5-2026-04-24'


def make_client():
    key = os.environ.get('QWEN_PIXEL_API_KEY')
    if not key:
        # Compatibility with the already configured project client. Never log it.
        tree = ast.parse((ROOT / 'scripts/build_teacher_magicbrush.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == 'api_key' and isinstance(keyword.value, ast.Constant):
                        key = keyword.value.value
                        break
            if key:
                break
    if not key:
        raise RuntimeError('Set QWEN_PIXEL_API_KEY or configure the existing teacher client.')
    return AzureOpenAI(api_key=key, api_version='2023-07-01-preview',
        azure_endpoint='https://search.bytedance.net/gpt/openapi/online/multimodal/crawl',
        timeout=180, max_retries=3)


def image_block(path):
    payload = base64.b64encode(Path(path).read_bytes()).decode()
    with Image.open(path) as im:
        mime = Image.MIME.get(im.format, 'image/png')
    return {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + payload}}


def crop_image_block(path,box):
    with Image.open(path)as image:
        crop=image.convert('RGB').crop(tuple(box))
    stream=io.BytesIO();crop.save(stream,format='PNG')
    payload=base64.b64encode(stream.getvalue()).decode()
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+payload}}


def api_messages(messages):
    """Map normalized tool-image feedback to the API's tool text + image message."""
    output = []
    pending = []
    for message in messages:
        role, content = message['role'], message['content']
        if role == 'assistant' and message.get('tool_calls'):
            calls = []
            for call in message['tool_calls']:
                call_id = 'call_' + str(len(output)) + '_' + str(len(calls))
                pending.append(call_id)
                function = call['function']
                calls.append({'id': call_id, 'type': 'function', 'function': {
                    'name': function['name'], 'arguments': json.dumps(function['arguments'])}})
            output.append({'role': role, 'content': content or None, 'tool_calls': calls})
        elif role == 'tool':
            call_id = pending.pop(0)
            if isinstance(content, list):
                text = '\n'.join(b['text'] for b in content if b['type'] == 'text')
                output.append({'role': 'tool', 'tool_call_id': call_id, 'content': text})
                output.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Actual image returned by the preceding view_image tool.'}
                ] + [image_block(b['image']) for b in content if b['type'] == 'image']})
            else:
                output.append({'role': 'tool', 'tool_call_id': call_id, 'content': content})
        elif isinstance(content, list):
            output.append({'role': role, 'content': [image_block(b['image'])
                if b['type'] == 'image' else b for b in content]})
        else:
            output.append({'role': role, 'content': content})
    return output


def judge_edit(client, source, edited, instruction,strict_extent=False,target_box=None):
    prompt = '''Judge an image edit. Image 1 is the source; image 2 is the result.
Instruction: ''' + instruction + '''
Score instruction_adherence, preservation of unrelated content, and visual_quality from 1 to 5.
A program merely executing is not success. A no-op, an incorrect target region, or changing the entire scene for a local edit must fail.
Return JSON with instruction_adherence, preservation, visual_quality, pass (boolean), and a brief reason.'''
    if strict_extent:
        prompt+='''
Strict target-extent verification: the edit must cover the full requested visible object or explicitly named component. Blurring only a face does not satisfy a request to blur the entire person; brightening only a small face patch does not satisfy brightening the person. If the instruction names clothing, color changes must preserve skin and unrelated objects. Uniform pixels may legitimately remain unchanged, but a small partial edit must not pass as a complete object edit.
Also return target_extent_score (1-5) and target_extent_complete (boolean). Set target_extent_complete false for partial-target shortcuts. Evaluate against the actual visible source and result, not the model's claims.'''
    content=[{'type':'text','text':prompt},image_block(source),image_block(edited)]
    if target_box is not None:
        content.extend([{'type':'text','text':
            'Annotation identifies the intended target instance at source pixel box '+json.dumps(target_box)+
            '. This box identifies the instance, not an edit mask: background and occluding objects inside it must still be preserved. Other instances of the same category outside it are not requested targets. The next two images are the source and result crops of that same annotated region; use them to inspect actual changes while retaining full-scene preservation checks.'},
            crop_image_block(source,target_box),crop_image_block(edited,target_box)])
    if strict_extent:
        differences,difference_block=difference_evidence(source,edited,target_box)
        content.extend([{'type':'text','text':
            'Additional measured pixel evidence: '+json.dumps(differences)+
            '. The next image shows absolute RGB source/result differences multiplied by a FIXED factor of 4; black means unchanged. '
            'This diagnostic helps distinguish a true no-op from a localized subtle edit and helps locate spill. It is not a target mask, a generated reference, or evidence of good appearance by itself. '
            'Judge actual edit strength and visual quality from the unamplified source/result; keep the same full-visible-target and preservation requirements. '
            'Uniform, clipped white, or pure black pixels can remain identical under a valid brightness/blur transform. Do not call an edit a literal no-op if a substantial measured change is visible on the intended object, but still fail changes that are too weak, incomplete, on the wrong object, or spill into unrelated content.'},difference_block])
    response = client.chat.completions.create(model=MODEL,
        messages=[{'role': 'user', 'content': content}],
        response_format={'type': 'json_object'}, reasoning_effort='low',
        max_completion_tokens=4096)
    text = (response.choices[0].message.content or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    result = json.loads(text)
    result['rubric_version']='full_visible_target_v1'if strict_extent else'legacy_edit_v1'
    result['annotated_target_box']=target_box
    if strict_extent:
        result['difference_evidence']=differences
        result['judge_evidence_version']='aligned_rgb_delta_v1'
    return score_judgment(result)
