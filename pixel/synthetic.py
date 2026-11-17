"""Procedural pixel scenes with exact palette/grid supervision."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Iterable

from PIL import Image, ImageDraw

from .palette import palette_grid_text, quantize_image

COLORS = {
    'red': (220, 40, 40),
    'green': (40, 160, 70),
    'blue': (40, 80, 220),
    'yellow': (240, 210, 60),
    'orange': (240, 140, 40),
    'purple': (150, 60, 200),
    'cyan': (40, 190, 200),
    'magenta': (220, 60, 180),
    'white': (245, 245, 245),
    'black': (30, 30, 30),
    'gray': (130, 130, 130),
    'brown': (140, 90, 50),
    'pink': (240, 150, 180),
}
BACKGROUNDS = ['white', 'black', 'gray', 'yellow', 'cyan', 'pink', 'green']
SHAPES = ['rectangle', 'ellipse', 'triangle', 'diamond', 'cross', 'ring']
EDIT_SHAPES = ['rectangle', 'ellipse', 'triangle', 'diamond']


def article(word: str) -> str:
    return 'an' if word[0].lower() in 'aeiou' else 'a'


def color_name(rgb: tuple[int, int, int]) -> str:
    for name, value in COLORS.items():
        if value == tuple(rgb):
            return name
    return 'unknown'


@dataclass
class Obj:
    shape: str
    color: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 inclusive

    def copy(self) -> 'Obj':
        return Obj(self.shape, self.color, self.bbox)


@dataclass
class Scene:
    size: int
    background: str
    objects: list[Obj] = field(default_factory=list)
    seed: int = 0

    def copy(self) -> 'Scene':
        return Scene(self.size, self.background, [o.copy() for o in self.objects], self.seed)

    def render(self) -> Image.Image:
        image = Image.new('RGB', (self.size, self.size), COLORS[self.background])
        draw = ImageDraw.Draw(image)
        for obj in self.objects:
            draw_obj(draw, obj, COLORS[self.background])
        return image

    def caption(self) -> str:
        parts = []
        for obj in self.objects:
            parts.append(f'{article(obj.color)} {obj.color} {obj.shape}')
        if len(parts) == 1:
            body = parts[0]
        elif len(parts) == 2:
            body = f'{parts[0]} and {parts[1]}'
        else:
            body = ', '.join(parts[:-1]) + f', and {parts[-1]}'
        return f'{body} on a {self.background} background'


def draw_obj(draw: ImageDraw.ImageDraw, obj: Obj, background: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = obj.bbox
    color = COLORS[obj.color]
    if obj.shape == 'rectangle':
        draw.rectangle([x0, y0, x1, y1], fill=color)
    elif obj.shape == 'ellipse':
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif obj.shape == 'triangle':
        draw.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color)
    elif obj.shape == 'diamond':
        draw.polygon([((x0 + x1) // 2, y0), (x1, (y0 + y1) // 2),
                      ((x0 + x1) // 2, y1), (x0, (y0 + y1) // 2)], fill=color)
    elif obj.shape == 'cross':
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        arm = max(1, (x1 - x0) // 4)
        draw.rectangle([cx - arm, y0, cx + arm, y1], fill=color)
        draw.rectangle([x0, cy - arm, x1, cy + arm], fill=color)
    elif obj.shape == 'ring':
        draw.ellipse([x0, y0, x1, y1], fill=color)
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            pad = max(1, min((x1 - x0) // 4, (y1 - y0) // 4))
            draw.ellipse([x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill=background)
    else:
        raise ValueError(obj.shape)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_bbox(rng: random.Random, size: int) -> tuple[int, int, int, int]:
    # Objects must cover enough of the grid that an all-background prediction
    # is not a strong local optimum.
    lo = max(2, size // 5)
    hi = max(lo + 2, size // 2)
    w = rng.randint(lo, hi)
    h = rng.randint(lo, hi)
    x0 = rng.randint(1, size - w - 2)
    y0 = rng.randint(1, size - h - 2)
    return x0, y0, x0 + w, y0 + h


def make_scene(rng: random.Random, size: int, n_objects: int | None = None) -> Scene | None:
    background = rng.choice(BACKGROUNDS)
    n = n_objects if n_objects is not None else rng.choice([1, 1, 2, 2, 3])
    colors = [c for c in COLORS if c != background]
    rng.shuffle(colors)
    scene = Scene(size=size, background=background, seed=rng.randrange(2 ** 31))
    for i in range(n):
        shape = rng.choice(SHAPES)
        color = colors[i]
        placed = False
        for _ in range(60):
            bbox = _random_bbox(rng, size)
            if all(not _overlaps(bbox, o.bbox) for o in scene.objects):
                scene.objects.append(Obj(shape, color, bbox))
                placed = True
                break
        if not placed:
            return None
    return scene


def make_pattern(rng: random.Random, size: int) -> Scene | None:
    """A scene-like object for stripes/checkerboards, rendered without Obj list."""
    kind = rng.choice(['stripes_h', 'stripes_v', 'checker'])
    colors = [c for c in COLORS if c not in {'white', 'black'}]
    rng.shuffle(colors)
    a, b = colors[0], colors[1]
    scene = Scene(size=size, background=a, seed=rng.randrange(2 ** 31))
    scene.pattern = (kind, a, b)  # type: ignore[attr-defined]
    return scene


def render_pattern(scene: Scene) -> Image.Image:
    kind, a, b = scene.pattern  # type: ignore[attr-defined]
    image = Image.new('RGB', (scene.size, scene.size), COLORS[a])
    draw = ImageDraw.Draw(image)
    if kind == 'checker':
        block = max(2, scene.size // 4)
        for y in range(0, scene.size, block):
            for x in range(0, scene.size, block):
                if ((x // block) + (y // block)) % 2:
                    draw.rectangle([x, y, x + block - 1, y + block - 1], fill=COLORS[b])
    elif kind == 'stripes_h':
        width = max(2, scene.size // 4)
        for y in range(0, scene.size, width * 2):
            draw.rectangle([0, y, scene.size - 1, y + width - 1], fill=COLORS[b])
    else:
        width = max(2, scene.size // 4)
        for x in range(0, scene.size, width * 2):
            draw.rectangle([x, 0, x + width - 1, scene.size - 1], fill=COLORS[b])
    return image


def pattern_caption(scene: Scene) -> str:
    kind, a, b = scene.pattern  # type: ignore[attr-defined]
    if kind == 'checker':
        return f'a {a} and {b} checkerboard'
    if kind == 'stripes_h':
        return f'horizontal {a} and {b} stripes'
    return f'vertical {a} and {b} stripes'


def _free_position(scene: Scene, rng: random.Random) -> tuple[str, tuple[int, int, int, int]] | None:
    size = scene.size
    slots = {
        'top left': (1, 1, size // 2 - 1, size // 2 - 1),
        'top right': (size // 2 + 1, 1, size - 2, size // 2 - 1),
        'bottom left': (1, size // 2 + 1, size // 2 - 1, size - 2),
        'bottom right': (size // 2 + 1, size // 2 + 1, size - 2, size - 2),
        'center': (size // 3, size // 3, 2 * size // 3, 2 * size // 3),
    }
    names = list(slots)
    rng.shuffle(names)
    for name in names:
        bbox = slots[name]
        if all(not _overlaps(bbox, o.bbox) for o in scene.objects):
            return name, bbox
    return None


def make_edit(scene: Scene, rng: random.Random) -> tuple[Scene, str, str] | None:
    """Return (target_scene, instruction, operation) or None."""
    if not scene.objects:
        return None
    op = rng.choice(['recolor', 'remove', 'add', 'move', 'flip', 'rotate'])
    target = scene.copy()
    size = scene.size
    if op in {'recolor', 'remove', 'move'}:
        idx = rng.randrange(len(scene.objects))
        obj = scene.objects[idx]
        if op == 'recolor':
            used = {o.color for o in scene.objects} | {scene.background}
            choices = [c for c in COLORS if c not in used]
            if not choices:
                return None
            new_color = rng.choice(choices)
            target.objects[idx].color = new_color
            return target, f'change the {obj.color} {obj.shape} to {new_color}', op
        if op == 'remove':
            target.objects.pop(idx)
            return target, f'remove the {obj.color} {obj.shape}', op
        # move
        dx, dy = rng.choice([(-3, 0), (3, 0), (0, -3), (0, 3)])
        x0, y0, x1, y1 = obj.bbox
        if x0 + dx < 1 or y0 + dy < 1 or x1 + dx > size - 2 or y1 + dy > size - 2:
            return None
        new_bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if any(_overlaps(new_bbox, o.bbox) for j, o in enumerate(scene.objects) if j != idx):
            return None
        target.objects[idx].bbox = new_bbox
        direction = 'right' if dx > 0 else 'left' if dx < 0 else 'down' if dy > 0 else 'up'
        return target, f'move the {obj.color} {obj.shape} {direction}', op
    if op == 'add':
        slot = _free_position(scene, rng)
        if not slot:
            return None
        position, bbox = slot
        used = {o.color for o in scene.objects} | {scene.background}
        choices = [c for c in COLORS if c not in used]
        if not choices:
            return None
        color = rng.choice(choices)
        shape = rng.choice(EDIT_SHAPES)
        target.objects.append(Obj(shape, color, bbox))
        return target, f'add {article(color)} {color} {shape} at the {position}', op
    if op == 'flip':
        return target, 'flip the entire image horizontally', op
    if op == 'rotate':
        return target, 'rotate the entire image 90 degrees clockwise', op
    return None


def apply_edit(scene: Scene, target: Scene, operation: str) -> Image.Image:
    if operation == 'flip':
        return scene.render().transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if operation == 'rotate':
        return scene.render().transpose(Image.Transpose.ROTATE_270)
    return target.render()


def pu_questions(scene: Scene, rng: random.Random) -> list[tuple[str, str]]:
    questions: list[tuple[str, str]] = []
    questions.append(('How many objects are in the image?', str(len(scene.objects))))
    questions.append(('What is the background color?', scene.background))
    shapes = [o.shape for o in scene.objects]
    unique_shapes = [s for s in set(shapes) if shapes.count(s) == 1]
    if unique_shapes:
        shape = rng.choice(unique_shapes)
        obj = next(o for o in scene.objects if o.shape == shape)
        questions.append((f'What color is the {shape}?', obj.color))
    leftmost = min(scene.objects, key=lambda o: (o.bbox[0], o.bbox[1]))
    if sum(1 for o in scene.objects if o.bbox[0] == leftmost.bbox[0]) == 1:
        questions.append(('Which object is leftmost? Answer with its color.', leftmost.color))
    all_colors = {o.color for o in scene.objects}
    missing = [c for c in COLORS if c not in all_colors and c != scene.background]
    if missing:
        questions.append((f'Is there a {rng.choice(missing)} object? Answer yes or no.', 'no'))
    if all_colors:
        questions.append((f'Is there a {rng.choice(sorted(all_colors))} object? Answer yes or no.', 'yes'))
    return questions
