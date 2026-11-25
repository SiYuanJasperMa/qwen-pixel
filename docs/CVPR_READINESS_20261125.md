# CVPR evidence gaps — 2026-11-25

## Submission timing

The official CVPR 2027 dates page lists paper registration Nov 10, 2026 AoE,
submission Nov 16, 2026 AoE, and supplementary materials Nov 23, 2026 AoE.
These dates have passed as of Nov 25 Beijing time. If a paper was already
submitted, the work can support subsequent review/revision; otherwise a new
main-conference submission requires a different cycle/venue. No acceptance
or future deadline is assumed here.

Source, checked directly: https://cvpr.thecvf.com/Conferences/2027/Dates

## Current evidence and missing work

| Requirement | Current evidence | Missing evidence |
|---|---|---|
| Reliable SFT interface | Native serialization/labels pass; reloaded single/three-photo pilots reproduce exact teacher RGB; 37-photo held-out results remain unreliable | General editing quality at scale; one-shot output/context budget alignment; preservation and structural edits |
| Data method improves learning | Continuing English RFT/TI production and a same-task/same-teacher paired builder exist | Completed 0/25/50% TI comparisons with equal assistant-token budgets; quality and rescue/enrichment ablations |
| General editing quality | Earlier frozen SFT runs have small custom-judge evaluations | Larger locked English evaluations across MagicBrush, UltraEdit, SciForma and MaskEdit English, with failure cases retained |
| RL improves an SFT model | English one-shot smoke completes 3 updates, reference SHA unchanged; same dev64 VLM passes rise 3→11 (10 gained/2 lost) | Deterministic repeat pre/post evaluation, second seed, stronger calibrated/official metrics; one legacy run is not a stable improvement claim |
| Correct visual measurement | Semantic/extent gates and target/preservation metric tools exist | Verified benchmark protocols, local/non-edit metrics, no-op controls and a blinded human quality audit |
| Source grounding | Image observations are verified by content hashes | Same-instruction/different-image controls and image-content swaps to test visual conditioning rather than instruction lookup |
| Scientific contribution | Executable pixel programs and a teacher verification pipeline are implemented | A clear advantage and method mechanism beyond assembling existing SFT/GRPO components |
| Reproducibility | Source-group splits, saved code/images and training logs exist | Frozen manifests, model/runtime hashes, complete compute/token reports and a release-ready data/source provenance record |

## Main experiments

1. Zero-shot Qwen3.8-27B, RFT SFT, mixed RFT+TI SFT, and SFT+RL on the same
   held-out inputs. Use at least two seeds for the main training comparisons.
2. Separate photo editing from scientific figure/infographic editing. Report
   add/remove/replace, recolor, local retouch and text/diagram operations rather
   than letting easy recolors dominate one aggregate number.
3. Report code/format validity separately from actual editing success. Include
   full-target coverage, non-edit preservation, target perceptual metrics and
   source/no-op baselines. Do not describe a custom judge as official scoring.
4. Audit whether target-informed references themselves satisfy instructions.
   Target similarity can conflict with instruction fidelity on noisy synthetic
   pairs. Inspect both retained and rejected TI examples.
5. Measure end-to-end inference latency, generated tokens, GPU footprint and
   Python execution time. Do not infer deployment efficiency from cheap code
   execution while ignoring language-model generation cost.

## Novelty and related work

VisPROG already uses language-generated visual programs and includes editing:
https://arxiv.org/abs/2211.11559 . A novelty claim must distinguish the learned
pixel-program policy, constrained runtime, data construction and verified
preservation/generalization behavior with experiments.

InstructPix2Pix is a relevant public-data and editing baseline:
https://huggingface.co/docs/diffusers/en/api/pipelines/pix2pix . Comparisons must
use matching benchmark splits, resolution and evaluation protocols. A reference
teacher that sees a target is an oracle construction condition, not a deployable
source-only editing baseline.

## Immediate priorities

Continue English RFT and independent TI production, align visible-answer and
inference budgets, and finish matched 0/25/50% TI one-shot SFT pilots. Repeat RL
pre/post and zero-shot evaluations with deterministic numeric settings and run
the second RL seed. Proceed to data-size scaling with measured image quality
and source/operation coverage. The current data plan is
`docs/IMAGE_EDIT_DATA_PLAN_20261125.md`.

## Evidence update: what is ready to test next

The matched256-task0/25/50%TI treatments have completed their approximately
one-million-assistant-token SFT budgets; all three are in deterministicdev64
evaluation. Two actual one-shot GRPO seeds125/126 have each completed3 nonzero
parameter updates, with frozen reference identity preserved. Deterministic
initial/post-RL comparison and the second seed'sdev64 evaluation are live.
Partial evaluation totals must not be presented as the final method result.

The next experimental unit is an editing capability, not a teacher brand or
dataset name. Report separate local-appearance, structural-object and
scientific/text/layout performance, then show whetherTI improves target fidelity
and usable task coverage at fixed tokens. A high-quality data selection study
and a same-sourceRFT/TI comparison are needed to distinguish better supervision
from merely repeating easy prompts. Study failure-rescueTI separately from
independentTI enrichment.

An agentic mask/crop-recovery pilot remains a debugging experiment. Do not make
the main paper a multi-step agent study unless that interface is explicitly
chosen later. The current main interface is one response emitting a pixel
program, followed by source-only Python execution. Compare image-only/image+RGB/
RGB-only input as an optional measured representation ablation; it does not
replace real held-out one-shot editing results.

Before a larger RL run, finish the matched pre/post evaluations, inspect all
rewarded incorrect edits and regressions, calibrate the judge against human and
benchmark measurements, and require enough semantically distinct successful
rollouts. Three updates prove engineering viability, not sustained learning.
