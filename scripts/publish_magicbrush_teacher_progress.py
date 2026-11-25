"""Publish the live MagicBrush teacher-progress dashboard to a static HF Space."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
OUT = ROOT / 'outputs/magicbrush_teacher_progress'
REPO = 'HuggingFriends/magicbrush-teacher-progress'
SKILL = Path('/home/tiger/.trae/skills/hf-static-pages/SKILL.md')


def token():
    match = re.search(r'hf_[A-Za-z0-9]{20,}', SKILL.read_text())
    if not match:
        raise RuntimeError('HF token not found in skill file')
    return match.group(0)


def main():
    os.environ['HF_TOKEN'] = token()
    api = HfApi(endpoint='https://hf-mirror.com')
    api.create_repo(repo_id=REPO, repo_type='space', space_sdk='static', exist_ok=True)
    readme = '''---
title: MagicBrush SFT Data Construction
emoji: 🏗️
colorFrom: purple
colorTo: blue
sdk: static
pinned: false
---

Live progress of MagicBrush teacher data construction.
'''
    readme_path = OUT / 'README.md'
    OUT.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme)
    api.upload_file(path_or_fileobj=str(readme_path), path_in_repo='README.md',
                    repo_id=REPO, repo_type='space', commit_message='init space')
    while True:
        try:
            subprocess.run([sys.executable, str(ROOT / 'scripts/render_magicbrush_teacher_progress.py')],
                           check=False, timeout=120)
            for name in ('index.html', 'progress.json'):
                api.upload_file(path_or_fileobj=str(OUT / name), path_in_repo=name,
                                repo_id=REPO, repo_type='space',
                                commit_message=f'progress update {name}')
            print(f'uploaded {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
        except Exception as exc:
            print(f'publish error: {type(exc).__name__}: {exc}', flush=True)
            time.sleep(30)
        time.sleep(60)


if __name__ == '__main__':
    main()
