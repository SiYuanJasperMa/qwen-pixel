# Qwen Pixel — active goal

Train Qwen to edit real RGB images as a coding agent, using the core PixelWorld workflow.

## User-facing behavior

Input: source photograph, editing instruction, and the same full-resolution RGB-readable image file.

Qwen chooses Python and image-viewing calls, reads and computes pixels, inspects or repairs results, and submits an edited PNG plus a matching uint8 RGB array. Keep this environment small and preserve native image dimensions.

## Active work

1. Collect actual teacher trajectories with calls, stdout/errors, returned images, and submitted RGB artifacts.
2. Judge edit quality and curate successful trajectories for assistant-action SFT.
3. Run Qwen3.5-9B Instruct LoRA SFT with real tool feedback in the conversation prefix.
4. Run online grouped agent RL in the same environment; verify actual rewards, gradients, and adapter updates.
5. Evaluate on held-out source photos and provide a visualization with complete real traces.
6. Expand to hundreds, then at least 10K accepted training trajectories once the pipeline and quality gates support scaling.

## Current pilot

- One replayed assistant-authored bus demonstration for format and overfit debugging.
- GPT-5.5 collection with eight concurrent tasks: 12-task pilot, extended to 80 source photos with 20 held out.
- A real-teacher SFT pilot and GRPO smoke are being run on the authorized H20 worker.
- Training/validation images are separated; final code alone is auxiliary data rather than a substitute for agent trajectories.

## Gates and limits

Require finite loss and genuine adapter changes, valid RGB submissions, semantic edit quality, and held-out autonomous execution. Equal-reward RL groups must not be reported as successful training updates. See docs/AGENT_PIPELINE.md for the explicit scale gates.

The active training mixture contains image editing only. No Qwen-Image, external image generator, learned renderer, physics engine, or palette-grid toy task is part of this path. Keep benchmark test data out of training and teacher supervision.

The older unified/palette proposal is retained only in docs/GOAL_LEGACY.md for history.
