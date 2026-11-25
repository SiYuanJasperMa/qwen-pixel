# Qwen Pixel-Edit

Train Qwen to edit real images by writing executable Python programs that compute the edited RGB pixels.

**Interface:** source image + English edit instruction → one Qwen response with evidence, plan and code → source-only Python execution → edited RGB/PNG.

- [Project page](https://siyuanjasperma.github.io/qwen-pixel/)
- [Live visualization](https://huggingfriends-qwen-pixel-live.static.hf.space/index.html)
- [Updated data construction plan](DATA_CONSTRUCTION_PLAN_V2.md)
- [Paper/data mixture references](docs/EDITING_TRAINING_DATA_REFERENCES_20261125.md)

## Current scope

Image editing only: natural photographs, English infographics and scientific diagrams. The main checkpoint is Qwen3.8-27B. Programs use PIL, NumPy and classical OpenCV; no pretrained editing or generation backend is invoked by the policy.

The six retained data sources are MagicBrush, UltraEdit region-based editing, applicable AnyEdit tasks, MaskEdit scene, MaskEdit English infographics, and SciForma Editing. InstructPix2Pix is disabled for new construction and future training snapshots; previous artifacts remain archived.

The V2 proposal targets 20K rigorously verified supervision rows, then data-size scaling based on measured validation gains. Its mixture weights and RFT/TI balance are experimental settings. Reliable unseen editing is still under investigation. The autonomous SFT goal is currently paused.

## Code

- `pixel/`: source/RGB execution, immutable observations, native protocol and validation.
- `train/run_pixel_sft.py`: native Qwen SFT and exact continuation state.
- `scripts/continuous_editing_teachers.py`: one-shot RFT/TI candidate generation.
- `scripts/build_paired_method_sft.py`: source/task/teacher-matched supervision comparison.
- `scripts/eval_local_qwen_oneshot.py`: one-shot editing evaluation.
- `scripts/run_agent_demo.py`: separate multi-turn protocol correctness diagnostics.
- `tests/`: format, label, identity and execution regressions.

Training images, generated data, model weights and private credentials are kept in the research workspace. Configure private API authentication locally; legacy proxy evaluation clients read `QWEN_PIXEL_PROXY_KEY` from the environment. The original exploratory overview is preserved in `docs/README_LEGACY.md`.

The project page exposes the live HF dashboard and a static snapshot, with the new data construction plan rendered for reading. Historical palette and agent experiments are archived separately and do not constitute official image-edit benchmark results.
