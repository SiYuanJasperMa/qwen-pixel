# Qwen Pixel — goal

Build a unified pixel-space post-training project for Qwen.

## Objective

Post-train a small Qwen vision-language model so that one checkpoint can:

1. **Understand pixels** — grounding, counting, color/position questions, and reading a palette-indexed raster.
2. **Generate images** — text-to-pixel (T2P) as a palette plus an index grid, following PixelBench.
3. **Edit images** — instruction plus source image to a new palette/grid, and high-resolution edits through executable PIL/NumPy code.
4. **Extend to video** — frame sequences and executable RGB programs, following PixelWorld.

The core deliverable is a **large, reproducible training set** and a working post-training recipe, not a single benchmark number.

## References

- **PixelBench** (`arXiv:2611.02540`): palette-indexed raster, T2P / I2P / pixel-understanding, Pixel Reconstruction Tuning. No learned tokenizer or decoder.
- **PixelWorld** (`/mnt/bn/embodied-lf3/masiyuan/PixelWorld`): coding agents as direct RGB world models; executable programs, exact PNG ↔ `uint32` validation, video tasks.
- **Qwen pixel edit milestone** (`/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke`): 9B Base LoRA, code-as-policy edits, grounding and QA; 92.2% real-photo region tint and 0/64 original MagicBrush semantic edits.

## Interface

Primary low-resolution interface:

```
PALETTE: [[r,g,b], ... up to 8 colors ...]
GRID:
<grid_size rows, each a string of palette indices>
```

Tasks:

| Task | Input | Output |
|---|---|---|
| `t2p` | text | palette + grid |
| `i2p` | image | palette + grid |
| `edit` | image + instruction | palette + grid |
| `pu` | palette + grid (text) | short answer |
| `ground` | image + referring expression | bbox JSON |
| `qa` | image + question | short answer |
| `code_edit` | image + instruction | executable PIL/NumPy code |
| `video` | text/image + instruction | frame sequence or executable program |

Low-resolution grids (16×16, 32×32) are the main scalable supervision. High-resolution and video use executable code so pixel values never have to be serialized as language tokens.

## Milestones

1. **Scaffold and format** — palette/grid codec, prompt templates, validator.
2. **Synthetic corpus** — procedurally generated T2P / I2P / PU / edit / video tasks with exact supervision. Target ≥ 100k records.
3. **Real-data conversion** — COCO/RefCOCO, MagicBrush, InstructPix2Pix, and the existing 19,776 code records converted to palette/grid or structured answers. Target ≥ 300k records.
4. **Smoke training** — Qwen3.5-9B-Base + LoRA, 16×16 first, then 32×32. Verify loss, parsing, and generated samples.
5. **Scale training** — mixed corpus, multi-GPU, 32×32, evaluate on held-out synthetic and real tasks.
6. **PixelBench-style evaluation** — palette/grid exactness, pixel accuracy, understanding accuracy, edit preservation.
7. **Video extension** — synthetic moving-scene sequences and PixelWorld-style executable programs.
8. **Public page** — reframe the existing site as Qwen Pixel with milestone results and galleries.

## Success criteria

- The palette/grid codec round-trips exactly on synthetic images and within quantization error on real images.
- A single 9B LoRA checkpoint can parse and produce valid palette/grid outputs at 16×16 and 32×32.
- Held-out synthetic T2P/I2P/PU accuracy is materially above an untrained Base baseline.
- Edit outputs preserve pixels outside the requested change better than a no-op baseline.
- The full data pipeline is reproducible from scripts and a manifest.

## Boundaries

- No external image/video generator, learned renderer, or physics engine in the generation path.
- Target images, masks, and bboxes never enter the model prompt.
- Public image pairs alone are not program supervision; converted targets must be explicitly defined operations.
- Keep failed and invalid results in the record; do not select only successful cases.
- Do not claim success from loss alone. Require parsed outputs, executed code, and held-out evaluation.
