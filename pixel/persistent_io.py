"""Pause writes during transient storage failures instead of losing results."""
import errno
import time


def retry_write(action):
    failures=0
    while True:
        try:return action()
        except OSError as exc:
            if exc.errno not in(errno.EIO,errno.ENOSPC,errno.EDQUOT,errno.ESTALE):raise
            failures+=1
            print('STORAGE_WRITE_WAIT',exc.errno,flush=True)
            time.sleep(min(60,2**min(failures,6)))


def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    def write():temp.write_text(text);temp.replace(path)
    retry_write(write)
