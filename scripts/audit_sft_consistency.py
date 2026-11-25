"""Compare gold-token loss, prompt logits, and generation before any retraining."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
from run_pixel_sft import encode, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=262144))
    model.eval()
    rows = [json.loads(l) for l in Path(args.data).open()][:3]
    records = []
    for row in rows:
        prompt = encode(processor, row, model.device)
        full = encode(processor, row, model.device, True)
        n = prompt['input_ids'].shape[1]
        gold = full['input_ids'][0, n:]
        with torch.no_grad():
            supervised = model(**full, use_cache=False)
            manual_loss = F.cross_entropy(supervised.logits[0,n-1:-1].float(), gold)
            prefill = model(**prompt, use_cache=False)
            prefix_diff = (supervised.logits[0,n-1].float()-prefill.logits[0,-1].float()).abs()
            # Last-prompt distribution versus the same token in teacher forcing.
            predicted = supervised.logits[0,n-1:-1].argmax(-1)
            generated = {}
            for cache in (False, True):
                ids = model.generate(**prompt, max_new_tokens=28, do_sample=False,
                    use_cache=cache,
                    eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
                generated[str(cache)] = processor.tokenizer.decode(ids[0,n:],skip_special_tokens=False)
        record = {'id':row['id'], 'fields':{k:list(v.shape) for k,v in full.items()},
            'reported_loss':float(supervised.loss),'manual_shifted_loss':float(manual_loss),
            'teacher_forced_token_accuracy':float((predicted==gold).float().mean()),
            'prompt_full_logits_max_difference':float(prefix_diff.max()),
            'prompt_full_logits_mean_difference':float(prefix_diff.mean()),
            'gold_text':processor.tokenizer.decode(gold.tolist(),skip_special_tokens=False),
            'teacher_forced_prediction_head':processor.tokenizer.decode(predicted[:28].tolist(),skip_special_tokens=False),
            'generated':generated}
        records.append(record)
        print(json.dumps(record),flush=True)
        Path(args.out).write_text(json.dumps(records,indent=2))
        del supervised, prefill


if __name__=='__main__':
    main()
