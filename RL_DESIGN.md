# Qwen Pixel-Edit · RL (GRPO) Design

## Goal

Push the SFT policy beyond the teacher-distillation ceiling using target-grounded
rewards. SFT imitates teacher code; RL optimizes the student directly against the
ground-truth edit result and preservation constraints.

## Setup

- Policy: Qwen3.8-27B + LoRA, initialized from the best SFT checkpoint.
- Prompt set: MagicBrush train (5k) + UltraEdit/SciForma accepted samples; 2-5k prompts is plenty.
- Group size: G=4 (smoke) → G=8 (main).
- Rollouts: sample with temperature 0.7-1.0, top-p 0.95, max 2,048 new tokens.
- One-shot format identical to SFT: system + source image + instruction → JSON with reasoning + code.

## Reward (local, cheap, no VLM judge in the inner loop)

| Component | Weight | Notes |
|---|---|---|
| format valid + code parses | +0.10 | JSON parses and `def edit(image)` exists |
| execution success | +0.20 | code runs on the source within timeout |
| target CLIP gain | +0.40 × clip_gain | cos(exec, target) - cos(source, target), clipped |
| target L1 gain | +0.20 × l1_gain | no_op_l1 - l1, clipped |
| preservation | +0.10 | L1 outside the changed-region mask must not increase |
| judge pass (optional) | +0.30 | only for a small calibration subset, not every rollout |

Total reward is clipped to [-1, 1]. A no-op gets 0 on the gain terms; a destructive
edit gets negative gain and negative preservation.

## Training

- GRPO: group-normalized advantages `A_i = (r_i - mean(r)) / (std(r) + eps)`.
- Loss: policy gradient with clipping (ratio 0.2) + KL to the SFT reference (beta 0.01-0.05).
- LoRA only (rank 16), same as SFT.
- Batch: 2-4 prompts per step, G rollouts each; 200-500 steps for the first run.
- Save every 50 steps; evaluate on MagicBrush dev32 + external dev.

## Infrastructure options

1. **HF generate batched rollout** (no new deps): 27B on one H20, batch G=4-8,
   ~1-2 min per prompt group. Slow but enough for a 1k-prompt smoke.
2. **vLLM + LoRA** (preferred for scale): TP=2 on 2 GPUs for rollout, 1 GPU for
   LoRA training, rest for evals. Requires installing vLLM in a separate venv to
   avoid disturbing the known-good SFT env.
3. **9B policy for recipe validation**, then transfer the recipe to 27B.

## Reward pitfalls

- **Code cheating**: hardcoded target pixels can maximize target similarity on the
  training source but fail on held-out sources. Use the flip probe and held-out
  source checks; penalize code with large literals or constant pixel writes.
- **No-op reward**: if the target is close to the source, doing nothing gets a
  positive CLIP similarity. Use the *gain* (relative to source), not raw similarity.
- **Judge-free hacking**: CLIP can be gamed by textures/colors; combine with L1 and
  preservation, and calibrate against the VLM judge on a subset.
- **Truncation**: cap max tokens and treat truncated outputs as format failures.

## Smoke plan (W2)

1. Build 200 prompts from MagicBrush train (not dev32).
2. Load the best SFT adapter; run G=4 rollouts with HF generate.
3. Execute + reward; log reward distribution and group variance.
4. One GRPO step; verify loss/gradients and that the policy changes.
5. Evaluate the updated adapter on dev32 (32 tasks) vs the SFT baseline.
6. If positive, scale to 1-2k prompts with vLLM.

## Success criterion

RL beats the best SFT checkpoint on MagicBrush dev32 (currently 21.9%) and does not
regress on UltraEdit/SciForma, with execution rate ≥ 85%.
