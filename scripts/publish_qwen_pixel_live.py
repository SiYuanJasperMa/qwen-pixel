"""Publish the current dashboard and keep it fresh without touching the old Space."""
import os
import argparse
import re
import sys
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from publish_magicbrush_teacher_progress import token


def upload_patterns(out):
    # Old galleries accumulate thousands of thumbnails. Upload the current
    # complete page and its referenced images, rather than rehashing all history.
    page = (out/'index.html').read_text()
    assets = sorted(set(re.findall(r'assets/[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp|svg)', page)))
    return ['index.html', 'README.md', '*.json', *assets]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    api = HfApi(endpoint='https://hf-mirror.com', token=token())
    out = ROOT / 'outputs/qwen_pixel_live'
    while True:
        try:
            subprocess.run([sys.executable, str(ROOT/'scripts/render_qwen_pixel_live.py')], check=True)
            api.upload_folder(repo_id='HuggingFriends/qwen-pixel-live', repo_type='space',
                folder_path=str(out), allow_patterns=upload_patterns(out),
                commit_message='Refresh Qwen Pixel-Edit progress')
            print('PUBLISHED', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        except Exception as exc:
            print('PUBLISH_ERROR', type(exc).__name__, str(exc)[:180], flush=True)
        if args.once:
            break
        time.sleep(120)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
