"""Recover absent loopback teacher transports without interrupting in-flight APIs."""
import fcntl
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROXY=ROOT.parent/'traex_api_proxy'
OUT=ROOT/'outputs/english_teacher_supervision'


def healthy(port):
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/health',timeout=3)as response:
            return json.load(response).get('ok')is True
    except Exception:
        return False


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'transport_supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            states={}
            for port,script in [(9584,'start_proxy.py'),(9588,'start_gpt62_proxy.py')]:
                state={'health':healthy(port)}
                if not state['health']:
                    try:
                        p=subprocess.run([sys.executable,str(PROXY/script)],capture_output=True,text=True,timeout=30)
                        state.update(restart_exit=p.returncode,health=healthy(port))
                    except Exception as exc:
                        state['restart_error']=type(exc).__name__
                states[str(port)]=state
            (OUT/'transport_status.json').write_text(json.dumps({'updated':time.time(),'ports':states}))
            print('TRANSPORT_HEALTH',json.dumps(states),flush=True)
            time.sleep(60)


if __name__=='__main__':main()
