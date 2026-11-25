"""Watch the exact RL and baseline processes, then evaluate the saved new policy."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rl_smoke_en_20261125'


def running(pidfile,marker):
    try:
        pid=int(pidfile.read_text().strip())
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return marker in command
    except (OSError,ValueError):return False


def main():
    watched=[(OUT/'train.pid','rl/run_oneshot_grpo.py')]+[
        (OUT/f'pre_shard{i}.pid','scripts/eval_local_qwen_oneshot.py')for i in range(4)]
    while any(running(path,marker)for path,marker in watched):
        print('WAIT_LIVE_PROCESSES',time.time(),flush=True);time.sleep(20)
    summary_path=OUT/'train/summary.json'
    if not summary_path.is_file():
        (OUT/'post_state.json').write_text(json.dumps({'state':'blocked_by_failed_rl_run','reason':'RL process exited without final summary'}))
        return
    summary=json.loads(summary_path.read_text())
    adapter=OUT/'train/final_adapter'
    if not(adapter/'adapter_model.safetensors').is_file():raise RuntimeError('Missing saved updated policy.')
    processes=[]
    for i in range(4):
        args=[sys.executable,'-u',str(ROOT/'scripts/eval_local_qwen_oneshot.py'),
            '--model',summary['configuration']['model'],'--adapter',str(adapter),
            '--tasks',str(ROOT/f'data/rl_smoke_en_20261125/eval_shard{i}.jsonl'),
            '--out',str(OUT/f'post/shard{i}'),'--limit','16','--max-new-tokens','4096']
        log=(OUT/f'post_shard{i}.log').open('w')
        process=subprocess.Popen(args,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i+3)},
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        log.close();processes.append(process)
        (OUT/f'post_shard{i}.pid').write_text(str(process.pid))
    (OUT/'post_state.json').write_text(json.dumps({'state':'running','pids':[p.pid for p in processes],
        'rl_updates':summary['optimizer_updates'],'completed_requested_updates':summary['completed_requested_updates']}))
    codes=[p.wait()for p in processes]
    (OUT/'post_state.json').write_text(json.dumps({'state':'completed'if not any(codes)else'failed','exit_codes':codes}))


if __name__=='__main__':main()
