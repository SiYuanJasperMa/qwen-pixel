"""One shared, source-only conversation contract for SFT, evaluation and RL."""
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'oneshot_native_source_first_v1'


def image_path(row):
    path = Path(row['image'])
    return path if path.is_absolute() else ROOT / path


def prompt_text(processor, row):
    content = []
    if row.get('image'):
        content.append({'type': 'image', 'image': str(image_path(row))})
    content.append({'type': 'text', 'text': row['instruction']})
    return processor.tokenizer.apply_chat_template(
        [{'role': 'system', 'content': row['system']},
         {'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_oneshot(processor, row, device, supervised=False, max_seq_len=None):
    text = prompt_text(processor, row)
    kwargs = {'text': [text], 'return_tensors': 'pt'}
    if row.get('image'):
        with Image.open(image_path(row)) as im:
            kwargs['images'] = [im.convert('RGB')]
    prompt = processor(**kwargs)
    n = prompt['input_ids'].shape[1]
    if supervised and 'target_token_ids' in row:
        # RL scores precisely the sampled tokens, with no decoding/re-tokenizing.
        target = torch.tensor([row['target_token_ids']], dtype=prompt['input_ids'].dtype)
        if not target.numel():
            raise ValueError('Empty sampled completion: ' + row['id'])
        result = dict(prompt)
        for key, value in list(result.items()):
            if isinstance(value, torch.Tensor) and value.shape == prompt['input_ids'].shape:
                extension = target if key == 'input_ids' else (
                    torch.ones_like(target) if key == 'attention_mask' else torch.zeros_like(target))
                result[key] = torch.cat([value, extension.to(value.dtype)], dim=1)
    elif supervised:
        kwargs['text'] = [text + row['answer'] + '<|im_end|>']
        result = processor(**kwargs)
        if not torch.equal(result['input_ids'][:, :n], prompt['input_ids']):
            raise ValueError('Prompt/answer token boundary mismatch: ' + row['id'])
    else:
        result = prompt
    if supervised:
        labels = result['input_ids'].clone()
        labels[:, :n] = -100
        if not (labels != -100).any():
            raise ValueError('No assistant tokens: ' + row['id'])
        result['labels'] = labels
    if max_seq_len and result['input_ids'].shape[1] > max_seq_len:
        return None
    return {k: v.to(device) for k, v in result.items()}
