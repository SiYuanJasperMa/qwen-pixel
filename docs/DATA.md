# Qwen Pixel data

## Row schema

```json
{
  "id": "synth_t2p_000001",
  "task": "t2p",
  "system": "...",
  "instruction": "T2P 32x32: a red circle on a white background",
  "image": null,
  "answer": "PALETTE: [[255,255,255],[255,0,0]]\nGRID:\n...",
  "meta": {"grid_size": 32, "palette_size": 2, "source": "synthetic_v1"}
}
```

Image-conditioned rows use a relative `image` path under the dataset root. Text-only rows use `"image": null`. Answers are either palette/grid text, a short string, bbox JSON, or executable code.

## Sources

### Synthetic (exact supervision)

- Shapes, colors, counts, positions, letters, digits, simple icons, patterns.
- T2P captions generated from the scene program.
- I2P targets from the same scene at 16×16 / 32×32 / 64×64.
- Edit operations: recolor, add, remove, move, flip, rotate, resize.
- PU questions: count, color, leftmost/rightmost, above/below, background.
- Video: moving shapes across frames; later.

Synthetic data is unlimited and exact. It is the main smoke and format-validation corpus.

### Real still images

- COCO / RefCOCO: captions for T2P, quantized images for I2P, referring expressions for grounding, and counts/colors for QA.
- MagicBrush: source/target pairs for low-resolution palette edits, plus original semantic-edit targets for evaluation.
- InstructPix2Pix: source/instruction/target triples for palette edits.
- Existing `qwen_image_edit_smoke` records: execute the saved program to obtain the target image, then quantize.

### PixelBench

If the official PixelBench data is released, import it directly. Otherwise reproduce its interface: up to eight colors, palette + index grid, T2P / I2P / pixel understanding at multiple resolutions.

### Video

- PixelWorld benchmark inputs are evaluation-only. Do not train on target videos or scores.
- Synthetic moving scenes and explicitly defined video edits provide training supervision.
- High-resolution video uses executable programs; low-resolution video uses per-frame grids.

## Conversion rules

1. Resize to the target grid size with LANCZOS.
2. Quantize to at most 8 colors without dithering.
3. Sort palette entries by frequency descending, then RGB ascending.
4. Reject grids with out-of-range indices or wrong dimensions.
5. Keep source image and target grid; never put the target grid in the prompt.
6. Record the operation and source dataset in `meta`.

## Splits

- Train / val / test split by scene, not by row, so the same scene cannot leak across splits.
- Held-out synthetic scenes use different seeds and object combinations.
- Real-image splits use COCO image IDs, not captions.

## Size targets

| Stage | Records | Purpose |
|---|---:|---|
| Format smoke | 2k | codec + trainer sanity |
| Smoke | 20k | 200–500 steps, 16×16 |
| Scale v1 | 200k | 32×32, mixed synthetic + real |
| Scale v2 | 1M+ | add more real data and video |

## Storage

- Images: PNG, 256×256 or the native grid render.
- Rows: JSONL, one file per split.
- Palette/grid targets: text in JSONL; no dense pixel arrays in language tokens.
- Manifest: source counts, seeds, code commit, and file hashes.
