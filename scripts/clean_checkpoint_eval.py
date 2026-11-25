"""Evaluate each completed checkpoint on fixed non-training development photos."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
    checkpoint=Path(args.checkpoint).resolve()
    if not(checkpoint/'run_state.json').exists():raise ValueError('Checkpoint is not complete.')
    state=json.loads((checkpoint/'run_state.json').read_text())
    training_data=Path(state['arguments']['data'])
    tasks=training_data.parent/'development_tasks.jsonl'
    if not tasks.exists():raise ValueError('Missing frozen development tasks.')
    output=checkpoint.parent.parent
    frequency=state['arguments']['save_every']
    manifest=training_data.parent/'manifest.json'
    data_metadata=json.loads(manifest.read_text())if manifest.exists()else{}
    profile='integer_edit'if data_metadata.get('geometry_format')=='integer_crop_0_1000'else'clean_edit'
    # Keep simultaneous experiments from loading several decoders onto the
    # same GPU. Queue checkpoints of each matched pilot on its own free card.
    dedicated={'sft_clean_mask_v4_raw':'3','sft_attention_pilot_v1':'4',
               'sft_refinement_pilot_v1':'0','sft_integer_pilot_v1':'7','sft_extended_v1':'1'}
    gpu=dedicated.get(output.name,str(1+(args.step//frequency)%2))
    lock_path=ROOT/'outputs/sft_audit'/f'evaluation_gpu_{gpu}.lock'
    lock_path.parent.mkdir(exist_ok=True)
    lock=lock_path.open('a')
    print('WAIT_EVALUATION_GPU',gpu,flush=True)
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    print('EVALUATION_GPU_ACQUIRED',gpu,flush=True)
    command=['/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python',
        'scripts/eval_agent_tasks.py','--model','/mnt/bn/embodied-lf3/masiyuan/robojudge_v4q/models/Qwen3.5-9B',
        '--adapter',str(checkpoint),'--tasks',str(tasks),
        '--out',str(output/f'development_{args.step}'),
        '--max-steps','8','--max-new-tokens','1800','--system-profile',profile,'--strict-extent']
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1'),check=True)
    subprocess.run([command[0],'scripts/render_sft_audit_page.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
