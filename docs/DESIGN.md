# Qwen Pixel design

## 1. Why palette-indexed rasters

PixelBench shows that a language model can generate and read images without a learned tokenizer or decoder by emitting a small palette followed by a grid of palette indices. This is a natural fit for Qwen:

- No architecture change and no image generation head.
- The output is ordinary text, so standard assistant-token cross entropy works.
- 16×16 and 32×32 grids are short enough for efficient SFT; 64×64 is a later scale-up.
- The same checkpoint can read a raster and answer questions about it.

For high-resolution edits and video, executable PIL/NumPy code remains more efficient than serializing every pixel. Qwen Pixel therefore uses a hybrid interface:

| Resolution / medium | Interface |
|---|---|
| 16×16, 32×32 still images | palette + grid |
| high-resolution still edits | executable code |
| video | per-frame grids (low-res) or executable RGB program (PixelWorld style) |

## 2. Codec

```
PALETTE: [[255,255,255],[0,0,0],[255,0,0]]
GRID:
0000000000000000
0001111000000000
...
```

- At most 8 colors, so every index is one digit.
- Palette order is deterministic: frequency descending, then RGB ascending.
- Grid rows are newline-separated; each row has exactly `grid_size` digits.
- The codec rejects out-of-range indices and wrong row/column counts.

## 3. Tasks

| Task | Prompt | Supervision |
|---|---|---|
| `t2p` | `T2P 32x32: <caption>` | palette + grid |
| `i2p` | image + `I2P 32x32: reconstruct ...` | palette + grid |
| `edit` | image + `EDIT 32x32: <instruction>` | palette + grid |
| `pu` | `PU: palette ... grid ... <question>` | short answer |
| `ground` | image + `GROUND: <referring expression>` | bbox JSON |
| `qa` | image + `QA: <question>` | short answer |
| `code_edit` | image + `EDIT: <instruction>` | executable code |
| `video` | text/image + instruction | frame grids or program |

`pu` is text-only: the raster is serialized in the prompt. This directly tests whether the model can read the same representation it generates.

## 4. Model and training

- Base: `Qwen/Qwen3.5-9B-Base`, revision `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- LoRA on the language side, rank 16–32, assistant-token cross entropy.
- One model handles both text-only (`t2p`, `pu`) and image-conditioned (`i2p`, `edit`) rows.
- Start at 16×16 (256 grid tokens) to validate the format, then 32×32 (1,024 tokens), then 64×64 if context and compute allow.
- Multi-GPU DDP on 8×H20 after the single-GPU smoke.
- Mix synthetic and real data; keep a small general-instruction mixture to reduce forgetting.

## 5. Evaluation

### Generation / reconstruction

- `parse_rate`: valid palette + grid output.
- `palette_exact`: palette matches the target up to ordering.
- `grid_exact`: every index matches.
- `pixel_accuracy`: fraction of grid positions matching.
- `color_error`: mean RGB distance after palette matching.
- Optional perceptual metrics (CLIP/DINO) once the grids are rendered.

### Editing

- `pixel_accuracy` on the full grid.
- `changed_region_accuracy`: accuracy inside the source/target difference region.
- `preserve_accuracy`: accuracy outside that region.
- `no_op_gain`: improvement over returning the source grid unchanged.

### Understanding

- Exact answer accuracy for counts, colors, positions, and object labels.
- Bbox IoU ≥ 0.5 for grounding.

### Video

- Per-frame grid accuracy, temporal consistency, and PixelWorld-style program execution checks.

## 6. Boundaries

- The model never sees target grids, masks, or bboxes during generation.
- Converted image pairs are not automatically program supervision; each operation is explicitly defined.
- A high score on synthetic scenes is not evidence of open-world semantic editing.
- Failures remain in the evaluation denominator and in the public galleries.
