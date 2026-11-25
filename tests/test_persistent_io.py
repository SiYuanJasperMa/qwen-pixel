import errno
import pytest
from pixel import persistent_io


def test_storage_failure_keeps_result_until_write_recovers(monkeypatch):
    sleeps=[];calls=[]
    monkeypatch.setattr(persistent_io.time,'sleep',sleeps.append)
    def write():
        calls.append(1)
        if len(calls)<3:raise OSError(errno.EDQUOT,'quota')
        return 'persisted'
    assert persistent_io.retry_write(write)=='persisted'
    assert len(calls)==3 and sleeps==[2,4]


def test_unrelated_error_is_not_retried():
    def write():raise OSError(errno.EACCES,'permission')
    with pytest.raises(OSError):persistent_io.retry_write(write)
