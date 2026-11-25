"""Move only the verified, unstarted control waiter to free model capacity."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'outputs/agent_sft_native_feedback_control_20261125'
    if(folder/'started.json').exists()or(folder/'train').exists():
        raise ValueError('Control training already started; do not interrupt it.')
    used=int(subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True).strip())
    if used>=4096:raise ValueError('GPU5 has a loaded workload.')
    record=ROOT/'outputs/agent_sft_native_feedback_comparison_20261125/launch.json'
    metadata=json.loads(record.read_text())
    previous=next(p for p in metadata['processes']if p['condition']=='control')
    path=Path(f'/proc/{previous["pid"]}/cmdline')
    if path.exists():
        args=path.read_bytes().decode().split('\0')
        if not any(s.endswith('launch_native_feedback_compare.py')for s in args)or '--wait-pid'not in args:
            raise ValueError('Recorded process is not the expected unstarted waiter.')
        if args[args.index('--condition')+1]!='control':raise ValueError('Different process ownership.')
        os.kill(previous['pid'],signal.SIGTERM)
    log=(record.parent/'control_rerouted.log').open('w')
    command=[sys.executable,'-u',str(ROOT/'scripts/launch_native_feedback_compare.py'),
        '--condition','control','--gpu','5']
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
    log.close()
    previous['previous_waiter']={k:previous[k]for k in ('gpu','pid','wait_pid')}
    previous.update(gpu=5,pid=child.pid,wait_pid=None)
    record.write_text(json.dumps(metadata,indent=2))
    print(json.dumps(previous),flush=True)


if __name__=='__main__':main()
