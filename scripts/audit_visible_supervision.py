"""Audit verified image histories for divergent next-action supervision."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.image_observations import load_observation
from pixel.agent_protocol import parse_call


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(rows, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B')
    image_cache = {}
    groups = defaultdict(list)
    actions = Counter()
    representations = Counter()
    for row in rows:
        messages = []
        visible_images = []
        for message in row['messages']:
            normalized = {k: v for k, v in message.items() if k not in ('edit_instruction',)}
            content = message.get('content')
            if isinstance(content, list):
                normalized['content'] = []
                for block in content:
                    if block.get('type') != 'image':
                        normalized['content'].append(block)
                        continue
                    key = (block['image'], block.get('sha256'))
                    if key not in image_cache:
                        image = load_observation(block)
                        image_cache[key] = {'size': list(image.size),
                            'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
                    normalized['content'].append({'type': 'image', **image_cache[key]})
                    visible_images.append(image_cache[key])
            messages.append(normalized)
        mode = row.get('tool_image_mode', 'tool')
        if mode == 'user':
            mapped = []
            for message in messages:
                content = message.get('content')
                if message['role'] == 'tool' and isinstance(content, list):
                    mapped.append({**message, 'content': '\n'.join(b['text'] for b in content if b.get('type') == 'text')})
                    images = [b for b in content if b.get('type') == 'image']
                    if images:
                        mapped.append({'role':'user', 'content':[{'type':'text',
                            'text':'Actual image returned by the preceding view_image tool.'}]+images})
                else:
                    mapped.append(message)
            messages = mapped
        kwargs = dict(tools=row['tools'], tokenize=False, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        signature = digest_json({'native_prompt':prompt, 'images':visible_images, 'tool_image_mode':mode})
        target = row['target_message']
        calls = target.get('tool_calls', [])
        if target['role'] != 'assistant':
            raise ValueError('Expected a single next-assistant action: ' + row['id'])
        if calls:
            if len(calls) != 1:
                raise ValueError('Expected one structured tool action: '+row['id'])
            action = calls[0]['function']['name']
            representations['structured_native_template'] += 1
        elif isinstance(target.get('content'), str):
            action, _ = parse_call(target['content'])
            representations['raw_native_assistant_text'] += 1
        else:
            raise ValueError('Missing native assistant tool action: '+row['id'])
        actions[action] += 1
        if 'target_token_ids' in row:
            identity = row['target_token_ids']
        else:
            full = tokenizer.apply_chat_template(messages+[target], add_generation_prompt=False, **kwargs)
            if not full.startswith(prompt):
                raise ValueError('Native supervised text does not extend prompt: '+row['id'])
            before = tokenizer.encode(prompt, add_special_tokens=False)
            all_ids = tokenizer.encode(full, add_special_tokens=False)
            if all_ids[:len(before)] != before:
                raise ValueError('Native target token boundary mismatch: '+row['id'])
            identity = all_ids[len(before):]
            stop = tokenizer.convert_tokens_to_ids('<|im_end|>')
            if stop not in identity:
                raise ValueError('Missing complete native assistant target: '+row['id'])
            end = identity.index(stop)+1
            if tokenizer.decode(identity[end:], skip_special_tokens=False).strip():
                raise ValueError('Stop precedes complete native target: '+row['id'])
            identity = identity[:end]
        groups[signature].append({'id': row['id'], 'action': action,
                                  'target_signature': digest_json(identity)})
    duplicates = [items for items in groups.values() if len(items) > 1]
    divergent = [items for items in duplicates if len({r['target_signature'] for r in items}) > 1]
    return {'rows': len(rows), 'verified_observation_blocks': len(image_cache),
            'unique_visible_histories': len(groups), 'actions': dict(actions),
            'target_representations': dict(representations),
            'duplicate_history_groups': len(duplicates), 'divergent_supervision_groups': divergent,
            'canonicalization': 'native Qwen prompt text, verified RGB content/dimensions and tool-image mode; supervised token IDs through EOS',
            'native_chat_template_sha256': digest_json(tokenizer.chat_template),
            'limitation': 'Distinct token sequences can both be valid; flags require review. Native text/RGB grouping does not substitute for a complete processor-tensor audit.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.data.read_text().splitlines() if s.strip()]
    report = audit(rows)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'divergent_supervision_groups'}))
    print('divergent_supervision_groups', len(report['divergent_supervision_groups']))


if __name__ == '__main__':
    main()
