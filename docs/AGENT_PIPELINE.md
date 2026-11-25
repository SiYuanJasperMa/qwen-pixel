# Qwen Pixel agent pipeline

Current scope: real-photo image editing through Python RGB operations. No Qwen-Image, generation service, learned renderer, or understanding-task mixture.

## Environment and supervision

The source is both an attached image and a full RGB-readable file. The actor can call `python`, call `view_image`, and submit an edited PNG and matching uint8 RGB array with `finish`. Files persist between Python calls; variables do not. Outputs must retain the original dimensions, match pixel-for-pixel between PNG and array, and actually change pixels.

Teacher data records actual calls, actual stdout/errors, actual returned images, and final submission. Teacher generation receives only the source and instruction. A separate judge checks the source, instruction, and result. Execution success is insufficient for SFT inclusion: the judged edit must also pass.

Each supervised record contains the conversation prefix and its next assistant action. Qwen's native chat/tool template serializes it. All prefix tokens, including previous assistant turns and tool returns, are masked for that record. Repeating this for every assistant action trains the trajectory decisions without supervising environment feedback. Images only appear after they were actually returned; future images do not enter earlier prefixes.

The first bus example is an assistant-authored demonstration with genuinely replayed feedback, separately marked from API teacher rollouts. Its one-task SFT is an overfit/pipeline check, not an evaluation of generalization.

## Running entry points

- `scripts/collect_agent_teachers.py`: GPT-5.5 teacher trajectories, eight concurrent tasks, retry-capable existing API client, per-task persistence and resume, image-disjoint splits, judged-pass export.
- `scripts/audit_agent_dataset.py`: split overlap, referenced-image availability, and supervised action coverage.
- `train/run_pixel_sft.py`: LoRA SFT; supports both older code rows and multi-image trajectory prefixes.
- `scripts/run_agent_demo.py`: autonomous Qwen execution and complete trajectory recording.
- `scripts/eval_agent_tasks.py`: full autonomous held-out rollouts and independent edit judging.
- `rl/run_agent_grpo.py`: online grouped rollouts in the same environment, semantic outcome rewards, group-normalized advantages, clipped token ratios, and a frozen-base KL reference.

RL scores the actual sampled assistant token IDs, not text re-tokenized after generation. The implementation performs one optimizer update per collected group. Equal-reward groups are reported and skipped. Nonzero gradients and parameter deltas must pass before reporting a successful RL update. This is a small single-worker smoke implementation, not a distributed training engine.

The local environment is for trusted teacher/student experiments. It applies import and path checks and a Python execution timeout, but is not a production security sandbox for adversarial code.

## Scale gates

1. Real teacher pilot: at least 90% valid submission and 70% judged editing success, with no image overlap between train and validation.
2. SFT: correct multimodal label masks, finite loss, nonzero LoRA updates, and autonomous tool execution on held-out images.
3. RL: varying actual rewards, finite nonzero gradients, changed adapter parameters, and before/after held-out evaluation. Running a reward loop without a weight update is insufficient.
4. First data expansion: about 80 distinct source photos, including 20 held-out photos, to expose format failures, truncation, and localization errors. At least 90% valid submissions and 60% judged editing success on this holdout are the initial gate for scaling this task family to hundreds/thousands of trajectories.

These gates establish pipeline readiness and a limited image-editing task family. They do not establish broad MagicBrush capability or statistically significant RL improvement. Object addition/replacement and other synthesis-heavy edits require separate evidence before expanding the scope.

Public benchmark test splits must never enter teacher SFT or RL training. Pilot held-out COCO photos are identified as a custom split, not reported as an official benchmark result.
