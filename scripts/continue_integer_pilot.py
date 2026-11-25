"""Run the matched integer-coordinate SFT only after exact-RGB smoke and data audits."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def wait_pid(pid):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    print('WAIT_OWNED_PROCESS',pid,flush=True);select.select([fd],[],[]);os.close(fd)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--builder-pid',type=int,required=True)
    parser.add_argument('--smoke-pid',type=int,required=True);args=parser.parse_args()
    wait_pid(args.builder_pid);wait_pid(args.smoke_pid)
    data=ROOT/'data/sft_integer_pilot_v1';out=ROOT/'outputs/sft_integer_pilot_v1'
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['trajectories']!=294 or not manifest['all_reference_RGB_exact']:raise ValueError('Integer data did not preserve all accepted RGB references.')
    smoke=ROOT/'outputs/sft_integer_smoke_v1'
    prefixes=json.loads((smoke/'prefixes.json').read_text())
    if len(prefixes)!=4 or not all(r.get('sampled_eos')and r.get('complete_call')and r.get('arguments_match')for r in prefixes):
        raise ValueError('Integer one-case full calls did not match.')
    task=json.loads((ROOT/'data/sft_integer_smoke_v1/tasks.jsonl').read_text())
    evaluation=json.loads((smoke/'rollout/summary.json').read_text())
    if evaluation['judge_pass']!=1:raise ValueError('Integer autonomous smoke did not pass strict judging.')
    expected=np.load(ROOT/'data/sft_integer_smoke_v1/executions'/task['id']/'edited_rgb.npy')
    actual=np.load(smoke/'rollout'/task['id']/'edited_rgb.npy')
    if not np.array_equal(expected,actual):raise ValueError('Integer autonomous smoke changed reference RGB.')
    subprocess.run([sys.executable,'scripts/audit_sft_dataset_budget.py','--data',str(data/'train.jsonl')],cwd=ROOT,check=True)
    (out/'smoke_gate.json').write_text(json.dumps({'calls_exact':4,'native_RGB_exact':True,'all294_reference_RGB_exact':True},indent=2))
    command=[sys.executable,'train/run_pixel_sft.py','train',
        '--model',str(ROOT.parent/'robojudge_v4q/models/Qwen3.5-9B'),
        '--data',str(data/'train.jsonl'),'--output',str(out/'train'),
        '--steps','1200','--accum','2','--rank','16','--lr','5e-5','--lora-scope','standard',
        '--geometry-weight','3','--nonzero-weight','1','--gradient-checkpointing',
        '--save-every','300','--eval-every','0','--max-seq-len','8192','--image-max-pixels','262144',
        '--checkpoint-hook','scripts/clean_checkpoint_eval.py']
    print('MATCHED_INTEGER_REPRESENTATION_PILOT_START',flush=True)
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',PYTHONUNBUFFERED='1'),check=True)


if __name__=='__main__':main()
