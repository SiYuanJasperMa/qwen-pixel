"""Check next-action predictions on actual teacher training prefixes."""
import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT))
from run_pixel_sft import encode, load
from pixel.agent_protocol import parse_call
from pixel.eval_manifest import output_budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=1800)
    parser.add_argument('--max-context',type=int,default=8192)
    parser.add_argument('--decode-policy',choices=['legacy','deterministic','fp32_state'],default='legacy')
    args = parser.parse_args()
    from pixel.decoding import configure_decoding,DecodePolicyModel
    configure_decoding(args.decode_policy)
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=262144))
    model.eval()
    if args.decode_policy!='legacy':model=DecodePolicyModel(model,args.decode_policy)
    rows = [json.loads(l) for l in Path(args.data).open()]
    if args.limit:
        rows=rows[:args.limit]
    results = []
    for row in rows:
        inputs = encode(processor, row, model.device)
        stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        attempts=[];stopped=False
        budgets=output_budgets(int(inputs['input_ids'].shape[1]),args.max_new_tokens,args.max_context)
        if not budgets:raise ValueError('Teacher prefix leaves no actor output budget: '+row['id'])
        for budget in budgets:
            with torch.inference_mode():
                ids=model.generate(**inputs,max_new_tokens=budget,do_sample=False,use_cache=True,eos_token_id=stop)
            tokens=ids[0,inputs['input_ids'].shape[1]:]
            stopped=len(tokens)>0 and int(tokens[-1])==stop
            attempts.append({'budget':budget,'tokens':len(tokens),'stopped':stopped})
            if stopped:break
        text = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        match = re.search(r'<function=([^>]+)>', text)
        function = row['target_message']['tool_calls'][0]['function']
        expected = function['name']
        result = {'id': row['id'], 'expected': expected,
                  'decode_policy':args.decode_policy,
                  'predicted': match[1] if match else None, 'text': text,
                  'sampled_eos':stopped,'generation_attempts':attempts}
        try:
            name, arguments = parse_call(text)
            result['complete_call'] = stopped
            result['arguments_match'] = stopped and arguments == function['arguments']
            if name == 'python':
                result['arguments_match'] = (stopped and arguments.get('code','').strip()
                                              == function['arguments'].get('code','').strip())
        except ValueError:
            result['complete_call'] = False
        results.append(result)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
