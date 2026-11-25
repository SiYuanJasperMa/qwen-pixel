# Qwen Pixel — image-edit training objective

Current data recipe: [DATA_CONSTRUCTION_PLAN_V2.md](DATA_CONSTRUCTION_PLAN_V2.md).
Its quotas are proposed experiment settings; the goal remains paused until resumed by the user.

## Current control: goal paused; InstructPix2Pix disabled

2026-11-25 user instruction: do not ingest or generate more InstructPix2Pix
samples, including RFT/TI or retries. Keep all existing generated data and old
experiments. New training snapshots exclude this source. Persistent policy:
`data/editing_source_policy.json`; source guards: `pixel/source_policy.py`.
The user has paused the SFT goal. Do not resume autonomous experiments until
the user resumes it. Previously submitted source requests may finish draining;
existing producer supervisors reload the source exclusion afterward.

## Latest user steering: 2026-11-25

Train a useful one-shot Qwen3.8-27B pixel-edit model on real image-edit tasks.
Continue English teacher construction, investigate SFT supervision and data
mixtures, and run genuine RL smokes with matched before/after evaluation.
Increase GPT-6.1/6.2-Sol concurrency outside Beijing working hours.
The earlier SFT-only restriction below is historical and superseded by this
instruction. The active SFT correctness goal remains unachieved; do not mark
the whole project complete after a small overfit or optimizer update.

Main interface: source image + instruction → one visible JSON response with
plan and Python → source-only program execution → edited RGB image.
Keep existing multi-turn agent correctness experiments as separate diagnostics.
No comprehension/COCO task is added to the main image-edit training mixture.

Current execution plan: docs/TRAINING_RECIPE_20261125.md and
docs/IMAGE_EDIT_DATA_PLAN_20261125.md.

Implemented2026-11-25: an additional supervised off-peakTI producer covers3746
English tasks with independent candidate IDs; night request caps now total
61-Sol56/62-Sol44 across production lines. The new process has a real daytime
`waiting_for_off_peak` heartbeat and will schedule after18:00 Beijing. Paired
0/25/50%TI one-shotSFT has completed its equal-token training budgets and is
evaluating64 development tasks. Two RL seeds each completed3 verified parameter
updates; matched deterministic before/after evaluation remains pending. Do not
infer model-quality wins before those comparisons finish.

Future training snapshots now protect candidate identity and immutable payloads.
Read `docs/CVPR_READINESS_20261125.md` for submission timing and required locked
cross-task evaluations, calibrated image metrics, data ablations and releases.

## Historical correctness investigation

Make Qwen agentic image-edit SFT work, with the same model-visible interface during training and inference.

Current scope: inspect and repair SFT formats, chat/tool templates, image returns, labels/loss masks, processor and adapter reload, then repeatedly test exact replay and independent RGB edits. Defer RL and bulk data expansion.

## Validation sequence

1. Audit training/inference initial prompts, native Qwen tool serialization, image preprocessing, and assistant-only shifted loss.
2. Overfit one real complete teacher trajectory and verify every next action after checkpoint reload.
3. Execute the task independently and compare actual RGB outputs.
4. Expand to a few distinct real editing tasks, verify training-prefix actions and independent execution, then test disjoint photos.
5. Keep testing until SFT follows the taught editing workflow without tool-call loops or unexplained regression.

## Current evidence

Serialization, shifted assistant-only loss, immutable observations, adapter updates, and saved-processor tensor parity have passed their audits. One trajectory passes 4/4 next actions; three trajectories pass 10/10 complete calls and independently reproduce all three teacher RGB outputs.

Unseen editing is still not reliable. The old ROI run passed 0/3 strict checks. The clean-mask v2 checkpoint at step 400 completed all eight development rollouts but passed only 1/8 full-visible-object editing checks. Training stopped at step 765 after reference-quality problems were found; checkpoint 400 is retained.

Only 40/64 annotation masks passed semantic coverage/exclusion review. These accepted sources were re-executed with corrected gamma brightening and weaker tint. Strict reference review retained 59 trajectories on 37 photos, with 236 assistant actions. Fresh v3 SFT is running for 1200 steps with a fixed-eight-image evaluation every 300 steps.

All 236 actual encoded rows fit both training and actor budgets: maximum context 3558, maximum assistant target 1408, versus limits 8192/4096. A six-prefix content-swap diagnostic on v2 increased polygon loss only 0.0224 nats on average, indicating a possible memorization shortcut. A separate 192-photo annotation-supervision pilot is being prepared, semantically filtered, and reference-reviewed before SFT on a separate GPU. No bulk model-generated teacher collection or RL is running.

The original eight development images and twenty independent COCO val2017 photos stay frozen and excluded from SFT. Geometry, masks, and evaluator boxes are never supplied to the inference model. The evaluator now receives measured aligned RGB differences to distinguish no-ops from subtle changes. Its unchanged strict gates rejected all four bad/no-op controls and accepted three real subtle edits; v2's saved outputs rejudge to 2/8, without regenerating the actor.

## Latest controlled SFT checks

The37-photo v3 run finished1200 steps. Checkpoints600,900,1200 each completed8/8 development submissions but passed0/8 strict edits. The113-photo standard-scope run finished1200 steps; its checkpoint600 completed7/8 and passed1/8. These results do not establish reliable editing.

Qwen3.5 has24 linear-attention and8 full-attention text layers. The new attention scope adapts context projections in all32 layers while freezing recurrent decay/output gates and MLP transformations. Its120-step one-photo smoke passes4/4 complete calls and exact arguments, independently reproduces native reference RGB, and passes strict judging. Actual autonomous-prefix tokens, image grids, pixel tensors, masks and multimodal type IDs match the teacher prefix exactly. A matched113-photo/1176-action attention run is being evaluated at300-step intervals.

Balanced semantic-error supervision was constructed on24 already-used training photos:47 actual bad-candidate/repair trajectories,188 correct actions. All47 initial bad images fail strict judging and all47 corrections pass. Bad actions are input history only, never loss targets. Corrections re-read a padded source region and compute the same native-source mask; image snapshots remain immutable. A1552-action dataset combines1176 normal actions with these188 repairs sampled twice. It is prepared and undergoing full budget audit, pending the matched comparison before the next training stage.

Evaluation GPU load is now spread across spare cards. Completed outputs are reused; only interrupted cases restart in fresh artifact directories. The GPU filesystem is authoritative for newly written run artifacts because host-side negative directory lookups can remain stale.

The1552-action repair-aware dataset passed all token-budget checks (maximum context5134, assistant1660). Fresh standard-scope SFT is running1600 steps on GPU6, with400-step development checks. A representation-only comparison now converts the same294 accepted edits to integer0-1000 ROI coordinates. Native RGB must remain exactly equal to every original reference. One real conversion and20 focused protocol tests passed; the120-step integer one-case SFT/independent execution gate is running on GPU5 before the matched113-photo pilot can start. No new training photos or learned image renderer were introduced.

The integer representation gate is now complete:4/4 full calls and arguments after120 steps, independent nativeRGB exactly equal to reference (427x640x3), strict judge pass. All294 integer-converted reference edits also exactly preserve originalRGB; the full1176-row integer budget audit is running before matched113-photoSFT. Standard113-photo checkpoint1200 completed8/8 development cases but still passed1/8; broad editing reliability remains unachieved. Keep goal active.

The6-photo generated-training-mask diagnostic has finished:5 literal masks parsed, meanIoU0.498, one action incomplete. Geometry is not sufficiently learned even on those training inputs. The earlier1200-step/accum2 run traversed1176 action samples only about2.04 times. A4000-step continuation from its completed adapter now runs on the exact same113-photo dataset, LR3e-5, warmup25, standardLoRA, checks every400 steps. Optimizer/sampler state restarts explicitly; this is not exact optimizer resume. Integer-format and repair-aware models continue their matched fixed8-image tests. No completion claim follows from improved training loss or mask fit alone.

Details and reproducible entry points: docs/SFT_CORRECTNESS.md.

No Qwen-Image, learned image renderer, external generation service, or toy palette task is part of this SFT path.
