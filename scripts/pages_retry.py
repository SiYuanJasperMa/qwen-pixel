"""Retry the Qwen Pixel Pages build after GitHub's build throttle clears.

The token is read from the existing git credential helper; it is never logged.
"""
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

REPO = 'SiyuanMaCS/qwen-pixel'
PIXELWORLD = '/mnt/bn/embodied-lf3/masiyuan/PixelWorld'
LOG = '/mnt/bn/embodied-lf3/masiyuan/qwen_pixel/outputs/pages_retry.log'


def log(message):
    line = f'{datetime.now(timezone.utc).isoformat()} {message}'
    print(line, flush=True)
    with open(LOG, 'a') as handle:
        handle.write(line + '\n')


def token():
    result = subprocess.run(
        ['git', 'credential', 'fill'], cwd=PIXELWORLD,
        input='protocol=https\nhost=github.com\n\n', text=True,
        capture_output=True, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return values['password']


def api(path, method='GET', data=None, password=None):
    request = urllib.request.Request(
        'https://api.github.com' + path, method=method,
        headers={'Authorization': 'Bearer ' + password,
                 'Accept': 'application/vnd.github+json',
                 'User-Agent': 'qwen-pixel'},
        data=json.dumps(data).encode() if data else None)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode() or '{}')
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode() or '{}')


def url_ok():
    try:
        request = urllib.request.Request('https://siyuanjasperma.github.io/qwen-pixel/',
                                         headers={'User-Agent': 'qwen-pixel'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    for attempt in range(1, 13):
        time.sleep(300)
        password = token()
        status, body = api(f'/repos/{REPO}/pages/builds', 'POST', {}, password)
        log(f'attempt={attempt} trigger={status} {str(body)[:120]}')
        time.sleep(30)
        status, body = api(f'/repos/{REPO}/pages/builds/latest', password=password)
        log(f'attempt={attempt} latest={status} {str(body)[:200]}')
        if url_ok():
            log('LIVE https://siyuanjasperma.github.io/qwen-pixel/')
            return
    log('GAVE_UP after 12 attempts')


if __name__ == '__main__':
    main()
