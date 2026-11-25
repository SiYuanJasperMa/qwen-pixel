"""Restart resumable, bounded TRAIN dataset ingestion after an unexpected exit."""
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/editing_growth_v1';folder.mkdir(exist_ok=True)
lock=(folder/'ingestion_supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
process=None
def stop(signum,frame):
    if process is not None and process.poll()is None:process.terminate()
    sys.exit(0)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
while True:
    print('START_RESUMABLE_INGESTION',flush=True)
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/ingest_editing_datasets.py')],cwd=ROOT)
    code=process.wait()
    print('INGESTION_RESTART_AFTER_EXIT',code,flush=True)
    time.sleep(30)
