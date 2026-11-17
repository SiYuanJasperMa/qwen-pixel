Public page: https://siyuanmacs.github.io/qwen-pixel/

# Qwen Pixel

Post-train Qwen as a pixel-space model: understand, generate, and edit images as palette-indexed rasters, with executable RGB programs for high-resolution edits and video.

This project extends the completed `qwen_image_edit_smoke` experiment. It follows two references:

- **PixelBench** (`arXiv:2611.02540`): a palette of up to eight RGB colors followed by a grid of palette indices; T2P, I2P, pixel understanding, and Pixel Reconstruction Tuning.
- **PixelWorld** (`../PixelWorld`): coding agents as direct RGB world models; executable programs and exact PNG ↔ `uint32` validation.

## Layout

```
qwen_pixel/
├── GOAL.md                 objective, milestones, boundaries
├── docs/DESIGN.md          interface, tasks, model and training plan
├── docs/DATA.md            data sources, schema, conversion rules
├── pixel/palette.py        palette/grid codec and validator
├── pixel/prompts.py        task prompt templates
├── pixel/synthetic.py      procedural scene generator
├── scripts/make_synthetic.py
├── scripts/convert_previous.py
├── scripts/build_mixture.py
├── train/run_pixel_sft.py
├── eval/eval_pixel.py
└── data/                   generated datasets (ignored by git)
```

## Quick start

```bash
PY=/mnt/bn/embodied-lf3/masiyuan/v2en_repro/.venv_known_good_sft/bin/python
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel

# 1. generate a small synthetic corpus
"$PY" scripts/make_synthetic.py --out data/synth_smoke_v1 --count 2000 --grid-size 16

# 2. validate the codec and dataset
"$PY" scripts/make_synthetic.py --out data/synth_smoke_v1 --count 2000 --grid-size 16 --validate-only

# 3. smoke train
CUDA_VISIBLE_DEVICES=0 "$PY" train/run_pixel_sft.py train \
  --model ../qwen_image_edit_smoke/models/Qwen3.5-9B-Base \
  --data data/synth_smoke_v1/train.jsonl \
  --val data/synth_smoke_v1/val.jsonl \
  --output outputs/smoke_v1 --steps 200 --rank 16
```

## Current status

Milestone 1–3 are in progress. The existing milestone-0 results (code-as-policy edits, grounding, QA) are documented in `../qwen_image_edit_smoke/RESULTS.md` and on the `qwen-pixel-edit` page.
