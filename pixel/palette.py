"""Palette-indexed raster codec used by Qwen Pixel.

A raster is a palette of at most eight RGB colors followed by a square grid of
palette indices. The text form is intentionally simple so a language model can
emit and read it without a learned tokenizer or decoder.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from PIL import Image

MAX_COLORS = 8


def _compact_palette(raw_palette: Sequence[int], indices: Sequence[int]) -> tuple[list[tuple[int, int, int]], list[int]]:
    counts = Counter(indices)
    used = sorted(counts, key=lambda i: (-counts[i], tuple(raw_palette[i * 3:i * 3 + 3])))
    remap = {old: new for new, old in enumerate(used)}
    palette = [tuple(int(v) for v in raw_palette[i * 3:i * 3 + 3]) for i in used]
    return palette, [remap[i] for i in indices]


def quantize_image(image: Image.Image, grid_size: int, max_colors: int = MAX_COLORS) -> tuple[list[tuple[int, int, int]], list[list[int]]]:
    """Resize and quantize an RGB image into a palette and a square index grid."""
    if not 1 <= max_colors <= 10:
        raise ValueError('max_colors must be between 1 and 10 so indices stay one digit')
    image = image.convert('RGB').resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    unique = Counter(pixels)
    if len(unique) <= max_colors:
        palette = sorted(unique, key=lambda c: (-unique[c], c))
        index = {c: i for i, c in enumerate(palette)}
        flat = [index[c] for c in pixels]
    else:
        quantized = image.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT,
                                   dither=Image.Dither.NONE)
        raw = quantized.getpalette() or []
        palette, flat = _compact_palette(raw, list(quantized.getdata()))
    grid = [flat[r * grid_size:(r + 1) * grid_size] for r in range(grid_size)]
    validate_palette_grid(palette, grid, grid_size)
    return palette, grid


def palette_grid_text(palette: Sequence[Sequence[int]], grid: Sequence[Sequence[int]]) -> str:
    palette_text = '[' + ','.join(f'[{r},{g},{b}]' for r, g, b in palette) + ']'
    rows = '\n'.join(''.join(str(int(v)) for v in row) for row in grid)
    return f'PALETTE: {palette_text}\nGRID:\n{rows}'


def parse_palette_grid(text: str) -> tuple[list[tuple[int, int, int]], list[list[int]]]:
    """Parse a model response into a palette and a square index grid."""
    cleaned = text.strip()
    match = re.search(r'PALETTE\s*:\s*(.+?)\s*GRID\s*:\s*(.+)$', cleaned, re.S | re.I)
    if not match:
        raise ValueError('Missing PALETTE or GRID section')
    palette_numbers = [int(v) for v in re.findall(r'-?\d+', match.group(1))]
    if len(palette_numbers) % 3:
        raise ValueError('Palette does not contain complete RGB triples')
    palette = [tuple(palette_numbers[i:i + 3]) for i in range(0, len(palette_numbers), 3)]
    if not 1 <= len(palette) <= MAX_COLORS:
        raise ValueError(f'Palette must have 1..{MAX_COLORS} colors, got {len(palette)}')
    if any(not 0 <= v <= 255 for color in palette for v in color):
        raise ValueError('Palette value outside 0..255')
    rows = []
    for raw_row in match.group(2).splitlines():
        row = re.sub(r'[^0-9]', '', raw_row)
        if not row:
            continue
        rows.append([int(v) for v in row])
    if not rows:
        raise ValueError('Empty grid')
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError(f'Grid is not square: {[len(r) for r in rows]}')
    validate_palette_grid(palette, rows, size)
    return palette, rows


def validate_palette_grid(palette: Sequence[Sequence[int]], grid: Sequence[Sequence[int]], grid_size: int | None = None) -> None:
    if not palette:
        raise ValueError('Empty palette')
    if len(palette) > MAX_COLORS:
        raise ValueError(f'Too many palette colors: {len(palette)}')
    for color in palette:
        if len(color) != 3 or any(not 0 <= int(v) <= 255 for v in color):
            raise ValueError(f'Invalid palette color: {color}')
    if not grid:
        raise ValueError('Empty grid')
    size = grid_size or len(grid)
    if len(grid) != size or any(len(row) != size for row in grid):
        raise ValueError('Grid dimensions do not match')
    for row in grid:
        for value in row:
            if not 0 <= int(value) < len(palette):
                raise ValueError(f'Grid index {value} outside palette of size {len(palette)}')


def render_palette_grid(palette: Sequence[Sequence[int]], grid: Sequence[Sequence[int]], scale: int = 8) -> Image.Image:
    size = len(grid)
    image = Image.new('RGB', (size, size))
    image.putdata([tuple(int(v) for v in palette[int(value)]) for row in grid for value in row])
    if scale != 1:
        image = image.resize((size * scale, size * scale), Image.Resampling.NEAREST)
    return image


def palette_exact(a: Iterable[Sequence[int]], b: Iterable[Sequence[int]]) -> bool:
    return {tuple(int(v) for v in c) for c in a} == {tuple(int(v) for v in c) for c in b}


def grid_exact(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return [list(map(int, row)) for row in a] == [list(map(int, row)) for row in b]


def pixel_accuracy(prediction: Sequence[Sequence[int]], target: Sequence[Sequence[int]]) -> float:
    if len(prediction) != len(target):
        return 0.0
    total = 0
    correct = 0
    for prow, trow in zip(prediction, target):
        if len(prow) != len(trow):
            return 0.0
        total += len(trow)
        correct += sum(int(a) == int(b) for a, b in zip(prow, trow))
    return correct / total if total else 0.0


def grid_to_rgb(grid: Sequence[Sequence[int]], palette: Sequence[Sequence[int]]) -> list[list[tuple[int, int, int]]]:
    return [[tuple(int(v) for v in palette[int(value)]) for value in row] for row in grid]
