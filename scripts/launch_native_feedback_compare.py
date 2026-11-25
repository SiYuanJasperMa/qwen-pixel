"""Run matched corrective SFT while respecting the recorded live GPU workload."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['feedback','control'])
    p.add_argument('--gpu',type=int);p.add_argument('--wait-pid',type=int)
    p.add_argument('--launch-pair',action='store_true');args=p.parse_args()
    if args.launch_pair:
        folder=ROOT/'outputs/agent_sft_native_feedback_comparison_20261125';folder.mkdir(exist_ok=True)
        if(folder/'launch.json').exists():raise ValueError('Inspect existing handles before restart.')
        handles=[]
        for condition,gpu,wait in [('feedback',2,None),('control',7,1840706)]:
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--condition',condition,'--gpu',str(gpu)]
            if wait:command+=['--wait-pid',str(wait)]
            with(folder/(condition+'.log')).open('w')as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            handles.append({'condition':condition,'gpu':gpu,'pid':child.pid,'wait_pid':wait})
        report={'worker':'5e5bed85e489ddcb','processes':handles,'steps':4500,'rows':300,
            'source_photos':37,'same_supervision_targets':True}
        (folder/'launch.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True);return
    if args.condition is None or args.gpu is None:p.error('Need condition and GPU, or launch-pair.')
    folder=ROOT/f'outputs/agent_sft_native_feedback_{args.condition}_20261125';folder.mkdir(exist_ok=True)
    if(folder/'started.json').exists():raise ValueError('Inspect recorded handle before restarting.')
    if args.wait_pid:
        while True:
            try:cmd=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
            except OSError:break
            if b'sft_37photos_exposure30.sh'not in cmd:break
            print('WAIT_LIVE_EXPOSURE_PIPELINE',args.wait_pid,flush=True);time.sleep(15)
    while True:
        used=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True).strip())
        if used<4096:break
        print('WAIT_EXISTING_GPU_WORKLOAD',args.gpu,used,flush=True);time.sleep(15)
    command=['bash','scripts/sft_native_feedback_compare.sh',args.condition]
    (folder/'started.json').write_text(json.dumps({'pid':os.getpid(),'gpu':args.gpu,'command':command,'time':time.time()}))
    code=subprocess.call(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu)})
    (folder/'exit.json').write_text(json.dumps({'exit_code':code}))
    raise SystemExit(code)


if __name__=='__main__':main()
