"""Move one verified owned evaluation to a free GPU, retaining completed cases."""
import argparse
import datetime
import json
import os
from pathlib import Path
import select
import signal
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--expected-out',type=Path,required=True)
    args=parser.parse_args();expected=args.expected_out.resolve()
    try:
        fd=os.pidfd_open(args.pid)
        argv=[a.decode()for a in Path(f'/proc/{args.pid}/cmdline').read_bytes().split(b'\0')if a]
    except (ProcessLookupError,FileNotFoundError):
        print('OLD_EVALUATION_ALREADY_TERMINAL; no restart.',flush=True);return
    if len(argv)<2 or not argv[1].endswith('scripts/eval_agent_tasks.py')or'--out'not in argv:
        raise ValueError('PID is not the expected evaluation command.')
    old_cwd=Path(os.readlink(f'/proc/{args.pid}/cwd'))
    old_out=Path(argv[argv.index('--out')+1]);old_out=old_out if old_out.is_absolute()else old_cwd/old_out
    if old_out.resolve()!=expected or old_cwd!=ROOT:raise ValueError('Owned evaluation identity does not match.')
    if (expected/'summary.json').exists():
        print('EVALUATION_ALREADY_COMPLETE; retain outputs.',flush=True);return
    os.kill(args.pid,signal.SIGTERM)
    if not select.select([fd],[],[],60)[0]:raise RuntimeError('Old evaluation has not stopped; do not start a duplicate.')
    os.close(fd)
    log=(expected/f'rerouted_gpu_{args.gpu}.log').open('a')
    new=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    event={'old_pid':args.pid,'new_pid':new.pid,'gpu':args.gpu,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reason':'Reduce confirmed concurrent decoder load on crowded GPU.',
        'completed_cases_reused':True,'incomplete_cases_restarted_in_fresh_attempt_directories':True}
    with (expected/'resource_reroutes.jsonl').open('a')as file:file.write(json.dumps(event)+'\n')
    print('EVALUATION_REROUTED',json.dumps(event),flush=True)


if __name__=='__main__':main()
