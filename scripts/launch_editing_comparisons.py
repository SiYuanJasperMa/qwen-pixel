"""Launch bounded matched one-shot experiments on the authorized H20 worker.

No platform job is created. Existing model processes are never evicted.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT.parent/'v2en_repro/.venv_known_good_sft/bin/python'
MODEL=ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    snapshot=args.snapshot.resolve()
    manifest=json.loads((snapshot/'manifest.json').read_text())
    for name,v in manifest['variants'].items():
        if hashlib.sha256((snapshot/name/'train.jsonl').read_bytes()).hexdigest()!=v['sha256']:
            raise ValueError('Frozen training data changed.')
    folder=ROOT/'outputs/paired_method_sft_20261125';folder.mkdir(exist_ok=True)
    with (folder/'launch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'launch.json').exists():
            raise ValueError('Experiments already launched; inspect exact recorded PIDs before restarting.')
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'],text=True)
        available={int(line.split(',')[0]):int(line.split(',')[1])for line in memory.splitlines()}
        if any(available[gpu]>4096 for gpu in range(7)):
            raise ValueError('Requested GPU has a loaded workload; refusing to evict it.')
        jobs=[]
        for gpu,name in [(4,'ti00'),(5,'ti25'),(6,'ti50')]:
            jobs.append((name,gpu,['bash','scripts/run_paired_method_sft.sh',name,str(gpu),str(snapshot)]))
        for name,gpu,adapter in [
            ('rl_initial_deterministic',0,'outputs/star_comparison_mixed_v1/train/checkpoint-1343'),
            ('rl_updated_deterministic',1,'outputs/rl_smoke_en_20261125/train/final_adapter'),
            ('zero_shot_deterministic',2,None)]:
            command=[str(PY),'-u','scripts/eval_local_qwen_oneshot.py','--model',str(MODEL),
                '--tasks','data/rl_smoke_en_20261125/tasks_eval64.jsonl','--out',str(folder/name),
                '--limit','64','--max-new-tokens','6144','--decode-policy','deterministic']
            if adapter:command+=['--adapter',adapter]
            jobs.append((name,gpu,command))
        jobs.append(('rl_seed126',3,['bash','scripts/run_second_rl_smoke.sh','3']))
        handles=[]
        for name,gpu,command in jobs:
            with (folder/(name+'.log')).open('w')as log:
                process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,
                    'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1',
                    'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            handles.append({'name':name,'gpu':gpu,'pid':process.pid,'command':command})
        report={'snapshot':str(snapshot),'model':str(MODEL),'task_budget':manifest['paired_tasks'],
            'assistant_token_budget_per_sft':1000000,'processes':handles,
            'data_ratios_are_row_fractions':True,'evaluation':'matched official MagicBrush dev64; custom image judge'}
        (folder/'launch.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
