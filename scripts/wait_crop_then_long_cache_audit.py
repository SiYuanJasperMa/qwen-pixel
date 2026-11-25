"""Wait for the recorded crop pipeline, then audit long source-conditioned decoding."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/agent_sft_crop_long_cache_20261125'


def main():
    OUT.mkdir(exist_ok=True)
    metadata=json.loads((ROOT/'outputs/agent_sft_crop_full_recovery_20261125/launch.json').read_text())
    pid=metadata['pid']
    while True:
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            live=b'sft_crop_recovery_execute.sh'in command
        except OSError:
            live=False
        if not live:break
        print('WAIT_VERIFIED_CROP_PIPELINE',pid,flush=True);time.sleep(15)
    while True:
        used=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True).strip())
        if used<4096:break
        print('WAIT_GPU7_LOADED_WORKLOAD',used,flush=True);time.sleep(15)
    args=[sys.executable,'-u',str(ROOT/'scripts/audit_cached_teacher_logits.py'),
        '--model',str(ROOT.parent/'robojudge_v4q/models/Qwen3.8-27B'),
        '--adapter',str(ROOT/'outputs/agent_sft_crop_feedback_paired_20261125/train/checkpoint-480'),
        '--data',str(ROOT/'data/sft_crop_correction_20261125_worker/train.jsonl'),
        '--id','clean_553533_1820009_blur_recovery_step_2','--tokens','1600',
        '--out',str(OUT/'logits.json')]
    result=subprocess.run(args,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'7'})
    (OUT/'status.json').write_text(json.dumps({'exit_code':result.returncode,'waited_for_pid':pid}))
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
