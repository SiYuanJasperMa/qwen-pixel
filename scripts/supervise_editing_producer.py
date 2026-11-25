"""Restart the resumable producer after an unexpected exit; keep one instance."""
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/editing_growth_v1'
if '--gpt62-only' in sys.argv[1:]:folder=folder/'gpt62'
if '--ti-only' in sys.argv[1:]:folder=folder/'ti_balanced_en_v1'
if '--out' in sys.argv[1:]:folder=Path(sys.argv[sys.argv.index('--out')+1])
folder.mkdir(parents=True,exist_ok=True)
lock=(folder/'supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
process=None
def stop(signum,frame):
    if process is not None and process.poll()is None:process.terminate()
    sys.exit(0)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
while True:
    print('START_RESUMABLE_PRODUCER',flush=True)
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/continuous_editing_teachers.py'),*sys.argv[1:]],cwd=ROOT)
    try:code=process.wait()
    except KeyboardInterrupt:
        process.terminate();raise
    print('PRODUCER_RESTART_AFTER_EXIT',code,flush=True)
    time.sleep(30)
