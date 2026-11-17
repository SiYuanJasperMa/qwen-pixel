"""Generate a synthetic Qwen Pixel corpus with exact palette/grid supervision."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pixel import prompts  # noqa: E402
from pixel.palette import palette_grid_text, parse_palette_grid, quantize_image  # noqa: E402
from pixel.synthetic import (  # noqa: E402
    Scene, apply_edit, make_edit, make_pattern, make_scene, pattern_caption,
    pu_questions, render_pattern,
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def split_for(rng: random.Random) -> str:
    value = rng.random()
    if value < 0.90:
        return 'train'
    return 'val' if value < 0.95 else 'test'


def write_scene(out: Path, split: str, scene_id: str, scene: Scene, is_pattern: bool) -> tuple[Path, str]:
    image = render_pattern(scene) if is_pattern else scene.render()
    dest = out / 'images' / split / f'{scene_id}.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest, relpath(dest)


def make_rows(out: Path, split: str, scene_id: str, scene: Scene, is_pattern: bool,
              rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    source_path, source_rel = write_scene(out, split, scene_id, scene, is_pattern)
    source = Image.open(source_path).convert('RGB')
    palette, grid = quantize_image(source, scene.size)
    answer = palette_grid_text(palette, grid)
    caption = pattern_caption(scene) if is_pattern else scene.caption()
    base = {'system': prompts.SYSTEM, 'meta': {'grid_size': scene.size,
            'palette_size': len(palette), 'source': 'synthetic_v1',
            'scene_id': scene_id, 'pattern': is_pattern}}
    for variant, text in enumerate([caption, f'pixel art of {caption}',
                                    f'a flat-color pixel image of {caption}']):
        rows.append({**base, 'id': f'{scene_id}_t2p_{variant:02d}', 'task': 't2p', 'image': None,
                     'instruction': prompts.t2p(text, scene.size), 'answer': answer,
                     'meta': {**base['meta'], 'caption': text}})
    rows.append({**base, 'id': f'{scene_id}_i2p', 'task': 'i2p', 'image': source_rel,
                 'instruction': prompts.i2p(scene.size), 'answer': answer,
                 'meta': {**base['meta'], 'caption': caption}})
    if not is_pattern:
        palette_text = '[' + ','.join(f'[{r},{g},{b}]' for r, g, b in palette) + ']'
        grid_text = '\n'.join(''.join(str(v) for v in row) for row in grid)
        for i, (question, expected) in enumerate(pu_questions(scene, rng)[:3]):
            rows.append({**base, 'id': f'{scene_id}_pu_{i:02d}', 'task': 'pu', 'image': None,
                         'instruction': prompts.pu(palette_text, grid_text, question),
                         'answer': expected,
                         'meta': {**base['meta'], 'question': question}})
        for i in range(3):
            edit = make_edit(scene, rng)
            if not edit:
                continue
            target_scene, instruction, operation = edit
            target_image = apply_edit(scene, target_scene, operation)
            target_path = out / 'targets' / split / f'{scene_id}_edit_{i:02d}.png'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_image.save(target_path)
            tpalette, tgrid = quantize_image(target_image, scene.size)
            rows.append({**base, 'id': f'{scene_id}_edit_{i:02d}', 'task': 'edit',
                         'image': source_rel, 'instruction': prompts.edit(instruction, scene.size),
                         'answer': palette_grid_text(tpalette, tgrid),
                         'meta': {**base['meta'], 'operation': operation,
                                  'instruction_text': instruction,
                                  'target_image': relpath(target_path)}})
    return rows


def validate(out: Path) -> dict:
    counts = {}
    for split in ['train', 'val', 'test']:
        path = out / f'{split}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            counts[row['task']] = counts.get(row['task'], 0) + 1
            if row['task'] in {'t2p', 'i2p', 'edit'}:
                palette, grid = parse_palette_grid(row['answer'])
                if len(grid) != row['meta']['grid_size']:
                    raise ValueError(f'Wrong grid size in {row["id"]}')
                if row['task'] in {'i2p', 'edit'}:
                    target = row['meta'].get('target_image') or row['image']
                    image = Image.open(PROJECT / target if not Path(target).is_absolute() else target).convert('RGB')
                    qpalette, qgrid = quantize_image(image, row['meta']['grid_size'])
                    if grid != qgrid:
                        raise ValueError(f'Answer does not match quantized target in {row["id"]}')
                    if len(palette) != len(qpalette):
                        raise ValueError(f'Palette size mismatch in {row["id"]}')
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=2000, help='approximate number of rows')
    parser.add_argument('--grid-size', type=int, default=16, choices=[8, 16, 32, 64])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out
    if args.validate_only:
        counts = validate(out)
        print(json.dumps({'validated': str(out), 'counts': counts}, indent=2))
        return

    rng = random.Random(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (out / f'{split}.jsonl').write_text('')
    handles = {split: (out / f'{split}.jsonl').open('a', buffering=1) for split in ['train', 'val', 'test']}
    rows_written = 0
    scene_index = 0
    target_rows = args.count
    while rows_written < target_rows:
        scene_index += 1
        scene_id = f'synth_{args.grid_size}_{scene_index:06d}'
        split = split_for(rng)
        is_pattern = rng.random() < 0.15
        scene = make_pattern(rng, args.grid_size) if is_pattern else make_scene(rng, args.grid_size)
        if scene is None:
            continue
        rows = make_rows(out, split, scene_id, scene, is_pattern, rng)
        for row in rows:
            handles[split].write(json.dumps(row) + '\n')
        rows_written += len(rows)
        if scene_index % 200 == 0:
            print(json.dumps({'scenes': scene_index, 'rows': rows_written}), flush=True)
    for handle in handles.values():
        handle.close()

    counts = validate(out)
    manifest = {'grid_size': args.grid_size, 'seed': args.seed, 'target_rows': target_rows,
                'scenes': scene_index, 'counts': counts,
                'files': {split: str((out / f'{split}.jsonl').relative_to(PROJECT)) for split in ['train', 'val', 'test']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
