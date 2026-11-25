# Qwen Pixel-Edit · ICML 2027 Plan

## One-line goal

Teach a general MLLM (Qwen3.8-27B) to edit real RGB images by writing **executable pixel programs**, one-shot, with no external tools at inference. Target: **ICML 2027** (abstract/full paper deadline around mid-to-late January 2027).

## Scope (locked)

- In scope: instruction-based image editing, RGB in → executable code → RGB out.
- Out of scope: image understanding / VQA, text→image generation, video, agentic tool loops.
- Base model: Qwen3.8-27B, LoRA SFT + GRPO RL.

## Paper thesis

> A general MLLM can be post-trained to express image editing as executable text programs; reasoning distillation from stronger teachers plus execution-grounded SFT and RL makes this interface reliable, interpretable and preservation-friendly.

## Contributions

1. Formulation: image editing as executable pixel programs in a pure-text token space.
2. Data engine: teacher reasoning distillation + execution verification + VLM judge filtering; 10k+ trajectories across MagicBrush / UltraEdit / SciForma.
3. Recipe: SFT data scaling, epochs, mixing ratios, reasoning-trace vs code-only ablation.
4. RL: GRPO with execution / pixel / preservation rewards on top of SFT; failure analysis and cross-dataset transfer.

## Experiment matrix

### Main
- zero-shot vs SFT(10k) vs SFT+RL.
- Benchmarks: MagicBrush dev (image-disjoint), UltraEdit dev94, SciForma dev55.
- Metrics: judge pass, execution rate, LPIPS / DINO / CLIP vs target, non-edit preservation.

### External baselines
- Qwen-Image-Edit (priority), InstructPix2Pix, MagicBrush model.
- Teacher upper bounds: GPT-6.1-Sol, GPT-5.6-Sol/Terra/Luna, GPT-5.5, DeepSeek-V4.1-Flash, Seed-2.1-Pro/Turbo (same prompt/exec/judge as student).

### Ablations
1. Data size 1k / 3k / 5k / 10k.
2. Epochs 0.5 / 1 / 2 / 3.
3. External mixing ratio 0% / 8% / 15% / 21%.
4. Reasoning trace vs code-only.
5. SFT vs SFT+RL; reward design (exec / exec+pixel / +judge).
6. RL group size, KL, rollouts per prompt.

### Analysis
- Failure taxonomy: exec fail / wrong object / incomplete edit / preservation violation.
- Per edit type: add / remove / replace / color / material.
- Reasoning length vs success.
- Cross-dataset transfer matrix.

## Timeline (9 weeks)

| Week | Dates | Goal |
|---|---|---|
| W1 | 11/20–11/26 | Data to 10k; v4/v5/v6 results; validation set 100+; v7 (10k, 2 epochs) |
| W2 | 11/27–12/3 | RL infra: vLLM rollout + local reward + GRPO; 1k-prompt smoke |
| W3–4 | 12/4–12/17 | RL at scale + reward/hparam ablations; main table |
| W5 | 12/18–12/24 | External baselines + failure analysis |
| W6 | 12/25–12/31 | Larger validation (200+), human eval, extra ablations |
| W7 | 1/1–1/7 | Final experiments, error bars, figures |
| W8 | 1/8–1/15 | Writing |
| W9 | 1/16–1/22 | Polish, reproducibility, submission |

## RL readiness gates

Start RL when, on a held-out set of 100+ images:
- execution rate ≥ 85–90%,
- judge pass rate ≈ 30–50% (mixed groups for GRPO),
- parse/format rate ≥ 95%.

Current: v3-cont-1200 on MagicBrush dev32 = 7/32 pass (21.9%), executed 21/32 (66%). Not yet.

## Current status (2026-11-19 night)

- **GPU paused at 23:08**: both workers were reallocated to videogen (minimax); all
  Qwen GPU training/eval stopped. Checkpoints: v4-900, v5-750, ultraedit-550; RFT
  had no checkpoint. GPU waiters were cleaned so nothing auto-restarts.
  Resume commands are in `outputs/RESUME_RUNBOOK.md`.
- SFT runs so far: v4 MagicBrush 5,082 (940/1,600), mixed v2 (done), v5 (769/2,000),
  ultraedit-only (590/800), RFT (48/1,600), v6/v9 not started. Best: v3-cont-1200 = 21.9%.
- **Official benchmark metrics implemented** (`scripts/eval_official_metrics.py`):
  L1/L2, PSNR, SSIM, LPIPS, CLIP-I, DINO, mask/preserve L1. MagicBrush dev32:
  student L1 0.0787 / LPIPS 0.213 / CLIP gain -0.0234; GPT-5.5 teacher 0.0619 / 0.190 / -0.0027;
  GPT-6.1 teacher 0.0628 / 0.192 / -0.0060; no-op L1 0.0447. Even teachers have
  negative CLIP gain: target-based pixel metrics are inherently hard for code editing.
- Official MagicBrush dev (528 tasks) downloaded and converted; SciFormaBench
  downloaded (generation benchmark, not editing).
- Data production continues on CPU/API: MagicBrush variants 3,053 / 5,000;
  target-informed 1,486; SciForma shards 1-8 at 241-315 judged each.
- v9 dataset ready: RFT 5,250 + target-informed 331 = 5,581 train / 604 val.
- Dashboard: https://huggingfriends-magicbrush-teacher-progress.static.hf.space/index.html
  (Experiment board, Data quality, Official benchmark metrics).

## Risks / fallbacks

- RL gives no gain → fall back to SFT scaling + data engine + analysis paper; retarget venue.
- 27B RL too heavy on 8×H20 → vLLM TP=2 + LoRA, or validate recipe on 9B and apply to 27B.
- Diffusion editors clearly better visually → position as interpretable / executable / preservation-friendly alternative; report gap honestly.
- Judge too slow for RL → local reward (LPIPS / DINO / CLIP + preservation), VLM judge only for calibration.
