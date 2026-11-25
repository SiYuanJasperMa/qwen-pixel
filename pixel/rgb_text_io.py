"""Vision-free Qwen language decoder and RGB-text one-shot prompting."""
import json
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from pixel.rgb_text_prompts import EDIT_SYSTEM, make_instruction


def make_prompt(tokenizer, task, mode='rgb_text'):
    text=make_instruction(task,mode)
    prompt=tokenizer.apply_chat_template([{'role':'system','content':EDIT_SYSTEM},
        {'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if '<|vision_start|>' in prompt or '<|image_pad|>' in prompt:
        raise ValueError('A pure-text prompt must not contain vision tokens.')
    return prompt


def encode_text_oneshot(tokenizer, task, device, mode='rgb_text', answer=None,
                        max_seq_len=None):
    """Identical text-only prefix for inference and assistant-only SFT."""
    prompt = make_prompt(tokenizer, task, mode)
    prefix = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    n = prefix['input_ids'].shape[1]
    result = prefix
    if answer is not None:
        if not answer.strip():
            raise ValueError('Empty assistant answer')
        result = tokenizer(prompt + answer + '<|im_end|>', return_tensors='pt',
                           add_special_tokens=False)
        if not torch.equal(result['input_ids'][:, :n], prefix['input_ids']):
            raise ValueError('RGB SFT/inference prompt token boundary mismatch')
        labels = result['input_ids'].clone()
        labels[:, :n] = -100
        result['labels'] = labels
    result = {k: v for k, v in result.items() if k in ('input_ids', 'attention_mask', 'labels')}
    if max_seq_len and result['input_ids'].shape[1] > max_seq_len:
        raise ValueError('RGB prompt exceeds context budget; never silently truncate pixels')
    return {k: v.to(device) for k, v in result.items()}


def load_text_decoder(model_path):
    full_config=AutoConfig.from_pretrained(model_path)
    config=full_config.text_config
    # Audit the real decoder's expected keys before loading any weights.
    with torch.device('meta'):
        skeleton=Qwen3_5ForCausalLM(config)
    expected=set(skeleton.state_dict())
    index=json.loads((Path(model_path)/'model.safetensors.index.json').read_text())['weight_map']
    supplied={k.replace('model.language_model.','model.',1) for k in index
              if k.startswith('model.language_model.') or k=='lm_head.weight'}
    if expected!=supplied:
        raise ValueError(f'Decoder mapping mismatch: missing={list(expected-supplied)[:5]}, extra={list(supplied-expected)[:5]}')
    del skeleton
    print('LANGUAGE_WEIGHT_KEYS_VERIFIED',len(expected),flush=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(model_path,config=config,
        key_mapping={r'^model\.language_model\.':'model.'},dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError('Some language weights were not loaded: '+str(info)[:1000])
    if any(name.endswith('visual') or '.visual.' in name for name,_ in model.named_modules()):
        raise ValueError('Vision module must not exist in the text-only model.')
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    print('PURE_LANGUAGE_MODEL_READY',model.__class__.__name__,sum(p.numel() for p in model.parameters()),flush=True)
    return model.eval(),tokenizer,{'class':model.__class__.__name__,'vision_modules':0,
        'language_weights_missing':len(info.get('missing_keys',[])),'language_weight_keys':len(expected)}
