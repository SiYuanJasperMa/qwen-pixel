"""Encode a trajectory prefix and supervise only its next assistant action."""
from pathlib import Path

import torch
from PIL import Image
from pixel.image_observations import load_observation

LABEL_POLICY='native_assistant_through_eos_v1'


def encode_trajectory(processor, row, device, supervised=False, max_seq_len=None):
    messages = row['messages']
    if row.get('tool_image_mode') == 'user':
        mapped = []
        for message in messages:
            content = message['content']
            if message['role'] == 'tool' and isinstance(content, list):
                text = '\n'.join(b['text'] for b in content if b.get('type') == 'text')
                mapped.append({**message, 'content': text})
                images = [b for b in content if b.get('type') == 'image']
                if images:
                    mapped.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': 'Actual image returned by the preceding view_image tool.'}
                    ] + images})
            else:
                mapped.append(message)
        messages = mapped
    kwargs = dict(tools=row.get('tools'), tokenize=False, enable_thinking=False)
    prompt_text = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, **kwargs)
    images = []
    for message in messages:
        if isinstance(message['content'], list):
            for block in message['content']:
                if block.get('type') == 'image':
                    images.append(load_observation(block))
    inputs = {'text': [prompt_text], 'return_tensors': 'pt'}
    if images:
        inputs['images'] = images
    prompt = processor(**inputs)
    if not supervised:
        result = prompt
    elif 'target_token_ids' in row:
        if 'target_message' in row and row['target_message'].get('role') != 'assistant':
            raise ValueError('Only an assistant action can be supervised.')
        # Preserve the exact sampled token sequence for on-policy RL scoring.
        result = dict(prompt)
        target = torch.tensor([row['target_token_ids']], dtype=prompt.input_ids.dtype)
        n = prompt.input_ids.shape[1]
        if not target.numel():
            raise ValueError('Empty sampled action.')
        for key, value in list(result.items()):
            if isinstance(value, torch.Tensor) and value.shape == prompt.input_ids.shape:
                extension = target if key == 'input_ids' else (
                    torch.ones_like(target) if key == 'attention_mask' else torch.zeros_like(target))
                result[key] = torch.cat((value, extension), dim=1)
        labels = result['input_ids'].clone()
        labels[:, :n] = -100
        result['labels'] = labels
    else:
        if 'target_text' in row:
            # RL must score the exact sampled assistant text, including its stop.
            full_text = prompt_text + row['target_text'] + '<|im_end|>'
        else:
            target = row['target_message']
            if target['role'] != 'assistant':
                raise ValueError('Only an assistant action can be supervised.')
            full_text = processor.tokenizer.apply_chat_template(
                messages + [target], add_generation_prompt=False, **kwargs)
        if not full_text.startswith(prompt_text):
            raise ValueError('Trajectory prompt is not a prefix of its training text.')
        inputs['text'] = [full_text]
        result = processor(**inputs)
        n = prompt.input_ids.shape[1]
        if not torch.equal(result.input_ids[:, :n], prompt.input_ids):
            raise ValueError('Trajectory token boundary mismatch: ' + row['id'])
        labels = result.input_ids.clone()
        labels[:, :n] = -100
        stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        ends=torch.nonzero(result.input_ids[0,n:]==stop,as_tuple=False).flatten()
        if not len(ends):raise ValueError('Missing native assistant stop token: '+row['id'])
        end=n+int(ends[0])+1
        trailer=processor.tokenizer.decode(result.input_ids[0,end:].tolist(),skip_special_tokens=False)
        if trailer.strip():raise ValueError('Assistant stop occurs before the complete target: '+row['id'])
        # Native templates append a conversation separator after assistant EOS.
        # Runtime generation ends at EOS, so this separator is not an action label.
        labels[:,end:]=-100
        if not (labels != -100).any():
            raise ValueError('No supervised action tokens: ' + row['id'])
        result['labels'] = labels
    if max_seq_len and result['input_ids'].shape[1] > max_seq_len:
        return None
    return {k: v.to(device) for k, v in result.items()}
