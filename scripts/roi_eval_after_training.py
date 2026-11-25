"""Run the strict held-out check as soon as the specific SFT process finishes."""
import os
from pathlib import Path
import select
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    try:
        descriptor=os.pidfd_open(301098)
        watcher=select.poll();watcher.register(descriptor,select.POLLIN);watcher.poll()
        os.close(descriptor)
    except ProcessLookupError:
        pass
    checkpoint=ROOT/'outputs/sft_roi_v1/train/checkpoint-1200'
    if not(checkpoint/'run_state.json').exists():raise RuntimeError('ROI SFT did not produce a completed checkpoint.')
    command=['/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python',
        'scripts/eval_agent_tasks.py','--model','/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B',
        '--adapter','outputs/sft_roi_v1/train/checkpoint-1200','--tasks','data/sft_roi_v1/tasks.jsonl',
        '--split','val','--out','outputs/sft_roi_v1/val_rollouts','--max-steps','8',
        '--max-new-tokens','1800','--system-profile','roi_edit','--strict-extent']
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1'),check=True)


if __name__=='__main__':main()
