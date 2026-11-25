"""Compare native cached decode logits with full teacher-forced SFT logits."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from run_pixel_sft import encode, load
from geometry_loss import geometry_indices
from pixel.decoding import configure_decoding, DecodePolicyModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--tokens', type=int, default=192)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    configure_decoding('deterministic')
    actor, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=262144))
    actor.eval()
    row = next(r for r in (json.loads(s) for s in Path(args.data).read_text().splitlines()) if r['id'] == args.id)
    prompt = encode(processor, row, actor.device)
    full = encode(processor, row, actor.device, supervised=True)
    n = prompt['input_ids'].shape[1]
    assert torch.equal(prompt['input_ids'], full['input_ids'][:, :n])
    labeled=full['labels'][0,n:]!=-100
    positions=torch.nonzero(labeled,as_tuple=False).flatten()
    if not len(positions)or not torch.equal(positions,torch.arange(len(positions),device=positions.device)):
        raise ValueError('Expected one contiguous assistant action, through its native stop.')
    target = full['input_ids'][0, n:n+len(positions)]
    count = min(args.tokens, len(target))
    if count <= 0:
        raise ValueError('No supervised tokens to compare.')
    gold = target[:count].cpu()
    full.pop('labels')
    for key in ['input_ids', 'attention_mask', 'mm_token_type_ids', 'token_type_ids']:
        if key in full:
            full[key] = full[key][:, :n + count]
    with torch.inference_mode():
        result = actor(**full, use_cache=False, logits_to_keep=count + 1)
        reference = result.logits[0, :-1].float().cpu()
    if reference.shape[0] != count:
        raise ValueError('Full-forward prediction positions do not match supervised tokens.')
    del result
    reference_logp = F.log_softmax(reference, dim=-1)
    reference_nll = -reference_logp[torch.arange(count), gold]
    pieces = [processor.tokenizer.decode([int(t)], skip_special_tokens=False) for t in gold]
    geometry = geometry_indices(pieces)
    report = {'id': row['id'], 'tokens': count, 'geometry_tokens': len(geometry),
              'scope': 'teacher-forced numerical audit; no image edit executed or counted as a success',
              'full_forward_mean_nll': float(reference_nll.mean()),
              'supervised_target_policy':'native action through EOS; no conversation trailer',
              'full_forward_per_token_nll':reference_nll.tolist(),
              'token_pieces':pieces,'full_forward_argmax_ids':reference.argmax(-1).tolist(),
              'conditions': {}}
    stop = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    def allow_gold(_batch, ids):
        index = ids.shape[-1] - n
        return [int(gold[index])]
    for policy in ['deterministic', 'fp32_state']:
        model = DecodePolicyModel(actor, policy)
        with torch.inference_mode():
            generated = model.generate(**prompt, max_new_tokens=count, do_sample=False, use_cache=True,
                eos_token_id=stop, prefix_allowed_tokens_fn=allow_gold,
                return_dict_in_generate=True, output_logits=True)
        emitted = generated.sequences[0, n:].cpu()
        if not torch.equal(emitted, gold):
            raise ValueError('The numerical audit did not use identical teacher tokens.')
        logits = torch.stack([x[0].float().cpu() for x in generated.logits])
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp[torch.arange(count), gold]
        kl = (reference_logp.exp() * (reference_logp - logp)).sum(dim=-1)
        agreement = logits.argmax(-1) == reference.argmax(-1)
        dtypes = sorted({str(layer.recurrent_states.dtype) for layer in generated.past_key_values.layers
                         if getattr(layer, 'recurrent_states', None) is not None})
        report['conditions'][policy] = {'recurrent_state_dtypes': dtypes,
            'mean_target_nll': float(nll.mean()), 'mean_kl_from_full': float(kl.mean()),
            'top1_agreement_with_full': float(agreement.float().mean()),
            'geometry_mean_target_nll': float(nll[geometry].mean()) if geometry else None,
            'geometry_mean_kl_from_full': float(kl[geometry].mean()) if geometry else None,
            'geometry_top1_agreement': float(agreement[geometry].float().mean()) if geometry else None,
            'argmax_ids':logits.argmax(-1).tolist(),
            'per_token_nll': nll.tolist(), 'per_token_kl': kl.tolist()}
        args.out.write_text(json.dumps(report, indent=2))
        print('CACHE_LOGIT_AUDIT', policy, json.dumps({k:v for k,v in report['conditions'][policy].items()
                                                    if not k.startswith('per_token')}), flush=True)
        del generated, logits, logp
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
