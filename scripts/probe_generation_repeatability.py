"""Probe greedy cached generation before/after another image prefix."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from run_pixel_sft import encode, load


def fingerprint(inputs):
    return {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                  'sha256': hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--preserve-recurrent-dtype', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(s) for s in Path(args.data).read_text().splitlines() if s.strip()]
    row = next(r for r in rows if r['id'] == args.id)
    donor = next(r for r in rows if r['meta']['original_scene_id'] != row['meta']['original_scene_id'])
    torch.manual_seed(42)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=262144))
    model.eval()
    stop = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    report = {'sample_id': row['id'], 'probe_tokens': args.tokens, 'greedy': True,
              'deterministic_algorithms': args.deterministic,
              'preserve_recurrent_dtype': args.preserve_recurrent_dtype,
              'recurrent_kernels': sorted({getattr(m.recurrent_gated_delta_rule, '__module__', '') + '.' + getattr(m.recurrent_gated_delta_rule, '__name__', '')
                                          for m in model.modules() if hasattr(m, 'recurrent_gated_delta_rule')}),
              'chunk_kernels': sorted({getattr(m.chunk_gated_delta_rule, '__module__', '') + '.' + getattr(m.chunk_gated_delta_rule, '__name__', '')
                                      for m in model.modules() if hasattr(m, 'chunk_gated_delta_rule')}),
              'scope': 'partial generation diagnostic; no action executed or counted successful',
              'steps': []}
    def run(sample, label):
        inputs = encode(processor, sample, model.device)
        cache_args = {}
        if args.preserve_recurrent_dtype:
            from pixel.generation_cache import fresh_precision_cache
            cache_args['past_key_values'] = fresh_precision_cache(model.config)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False,
                                       use_cache=True, eos_token_id=stop,
                                       return_dict_in_generate=True, **cache_args)
        ids = generated.sequences
        tokens = ids[0, inputs['input_ids'].shape[1]:].tolist()
        cache = generated.past_key_values
        state_dtypes = sorted({str(layer.recurrent_states.dtype)
                               for layer in cache.layers if getattr(layer, 'recurrent_states', None) is not None})
        item = {'label': label, 'id': sample['id'], 'input_fingerprints': fingerprint(inputs),
                'recurrent_state_dtypes': state_dtypes, 'sampled_eos': bool(tokens and tokens[-1] == stop),
                'output_token_ids': tokens, 'text': processor.tokenizer.decode(tokens, skip_special_tokens=False)}
        report['steps'].append(item)
        args.out.write_text(json.dumps(report, indent=2))
        print('PROBE_STEP', label, len(tokens), flush=True)
        return item
    first = run(row, 'first')
    repeat = run(row, 'immediate_repeat')
    run(donor, 'other_image')
    after = run(row, 'repeat_after_other_image')
    def compare(a, b):
        aa, bb = a['output_token_ids'], b['output_token_ids']
        return {'same_inputs': a['input_fingerprints'] == b['input_fingerprints'],
                'same_generation': aa == bb,
                'first_different_token': next((i for i, (x, y) in enumerate(zip(aa, bb)) if x != y),
                                             min(len(aa), len(bb)) if aa != bb else None)}
    report['immediate_comparison'] = compare(first, repeat)
    report['after_other_image_comparison'] = compare(first, after)
    # Run forward diagnostics afterward so an uncached forward cannot reset
    # hidden state between the generation calls under investigation.
    inputs = encode(processor, row, model.device)
    prefill_hashes = []
    with torch.inference_mode():
        for _ in range(2):
            logits = model(**inputs, use_cache=False).logits[0, -1].float().cpu()
            prefill_hashes.append(hashlib.sha256(logits.numpy().tobytes()).hexdigest())
    report['post_generation_prefill_repeat'] = {'hashes': prefill_hashes, 'same_logits': prefill_hashes[0] == prefill_hashes[1]}
    args.out.write_text(json.dumps(report, indent=2))
    print('GENERATION_REPEATABILITY_COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'steps'}), flush=True)


if __name__ == '__main__':
    main()
