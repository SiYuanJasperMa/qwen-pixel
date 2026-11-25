"""Run the full region/code prefix cohort after each matched checkpoint is saved."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    for scope in ['attention','standard']:
        out=ROOT/f'outputs/agent_sft_en_27b_37photos_{scope}_20261125'
        checkpoint=out/'train/checkpoint-400'
        while not(checkpoint/'run_state.json').exists():
            print('WAIT_SAVED_CHECKPOINT',scope,time.time(),flush=True);time.sleep(20)
        state=json.loads((checkpoint/'run_state.json').read_text())
        if state['step']!=400:raise ValueError('Unexpected checkpoint step.')
        result=out/'prefixes_broad118.json'
        if result.exists():continue
        command=[sys.executable,'-u',str(ROOT/'scripts/check_training_prefixes.py'),
            '--model',state['arguments']['model'],'--adapter',str(checkpoint),
            '--data',str(ROOT/'data/agent_sft_en_27b_37photos_20261125/prefixes_region_and_code.jsonl'),
            '--out',str(result),'--limit','0','--max-new-tokens','4096']
        log=(out/'prefixes_broad118.log').open('w')
        print('START_BROAD_PREFIX',scope,flush=True)
        subprocess.run(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpu},
            stdout=log,stderr=subprocess.STDOUT,check=True)
        log.close()
        print('BROAD_PREFIX_COMPLETE',scope,flush=True)


if __name__=='__main__':main()
