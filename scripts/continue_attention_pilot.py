"""Start the matched113-photo attention-scope SFT only after one-case proof."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-pid',type=int,required=True);args=parser.parse_args()
    try:fd=os.pidfd_open(args.smoke_pid)
    except ProcessLookupError:fd=None
    if fd is not None:
        print('WAIT_OWNED_ATTENTION_SMOKE',args.smoke_pid,flush=True);select.select([fd],[],[]);os.close(fd)
    smoke=ROOT/'outputs/sft_attention_smoke_v1'
    prefixes=json.loads((smoke/'prefixes.json').read_text())
    if len(prefixes)!=4 or not all(r.get('sampled_eos')and r.get('complete_call')and r.get('arguments_match')for r in prefixes):
        raise ValueError('Attention scope did not reproduce complete training calls.')
    evaluation=json.loads((smoke/'rollout/summary.json').read_text())
    if evaluation['complete']!=1 or evaluation['judge_pass']!=1:raise ValueError('One-case autonomous RGB edit did not pass.')
    task=json.loads((ROOT/'data/sft_attention_smoke_v1/tasks.jsonl').read_text())
    reference=ROOT/'data/sft_clean_mask_v4_raw/executions'/task['id']/'edited_rgb.npy'
    model=smoke/'rollout'/task['id']/'edited_rgb.npy'
    if not np.array_equal(np.load(reference),np.load(model)):raise ValueError('Attention smoke pixels differ from its reference.')
    data=ROOT/'data/sft_clean_mask_v4_raw/train_verified.jsonl'
    audit=json.loads((data.parent/'dataset_budget_audit.json').read_text())['summary']
    if audit['records']!=1176 or audit['context_overflows']:raise ValueError('Expected the same verified dataset as the standard-scope run.')
    out=ROOT/'outputs/sft_attention_pilot_v1'
    (out/'smoke_gate.json').write_text(json.dumps({'prefix_calls_exact':4,'native_rgb_exact':True,
        'strict_reference_edit_pass':True,'training_data':str(data),'fresh_base_initialization':True},indent=2))
    command=[sys.executable,'train/run_pixel_sft.py','train',
        '--model',str(ROOT.parent/'robojudge_v4q/models/Qwen3.5-9B'),
        '--data',str(data),'--output',str(out/'train'),'--steps','1200','--accum','2','--rank','16',
        '--lr','5e-5','--lora-scope','attention','--geometry-weight','3','--nonzero-weight','1',
        '--gradient-checkpointing','--save-every','300','--eval-every','0','--max-seq-len','8192',
        '--image-max-pixels','262144','--checkpoint-hook','scripts/clean_checkpoint_eval.py']
    print('MATCHED_ATTENTION_SCOPE_PILOT_START',flush=True)
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTHONUNBUFFERED='1'),check=True)


if __name__=='__main__':main()
