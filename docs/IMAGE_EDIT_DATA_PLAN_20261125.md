# One-shot Qwen Pixel-Edit: data and experimental design

> The current data construction recipe is [DATA_CONSTRUCTION_PLAN_V2.md](../DATA_CONSTRUCTION_PLAN_V2.md).
> This document retains historical experiments and evidence.

> Latest user policy,2026-11-25: InstructPix2Pix is disabled for new ingestion,
> RFT/TI, retries and future training snapshots. Existing samples/images and
> historical experiments stay archived. The SFT goal is paused; earlier source
> inventories below describe historical state and do not authorize resuming it.

The product objective is source image + English instruction → one Qwen response
containing an editing program → source-only execution → edited RGB. Generation
uses PIL/NumPy/classical OpenCV; learned editing/generation services are absent.
Data construction and evaluation must improve the rendered image, not merely
program syntax or answer likelihood. Preserve the separate agent diagnostics;
they are not the one-shot benchmark.

## What to construct

| Source family | Coverage to prioritize | What must be measured |
|---|---|---|
| MagicBrush / UltraEdit | Local recolor, removal, addition, replacement, retouch | Requested change, complete region, unrelated content preserved |
| AnyEdit | Broader edits, texture and lighting, global changes | Instruction fidelity and target consistency; do not impose a local-preservation rule on global tasks |
| MaskEdit scene / English infographic | Regional photo edits and text/layout edits | Localization, readable correct text, layout integrity |
| SciForma Editing | Scientific figure labels, curves, legends, style/layout | Exact requested symbol/text, preservation of numerical meaning and other figure elements |

These are data sources, not automatically official benchmarks. Record official
splits separately from our source-group holdouts. Scientific figures and natural
photos need separate scores. Do not advertise photorealistic unrestricted object
synthesis based on success with recolor or classical inpainting. Determine task
feasibility from retained failures as well as successful teacher examples.

Every task retains dataset/version/split, task ID, source group and native image
identity, source/target, instruction, edit category, teacher/version, construction
mode, API usage, visible response, code, rendered output and verification. Store
ambiguous categories as unknown instead of assigning a convenient operation.

## Two complementary teacher routes

**RFT:** one source + instruction teacher response, source-only execution and
verification. Retain instruction-correct complete edits. API failures are pending
technical work; semantic failure is not a training target.

**TI enrichment:** the teacher additionally sees the reference target, but emits
a source-executable program and source-grounded evidence. The student input
retains only source + instruction. Construct TI on RFT successes too, obtaining
paired RFT/TI supervision rather than selecting only unusually difficult tasks.

**TI rescue:** independently construct a target-informed answer after a genuine
RFT semantic failure. Keep it labeled separately from paired enrichment. This
tests additional task coverage; it is not the same treatment as improved answers
on already solvable tasks. Never include an earlier failed code as repair history
in the main one-shot interface.

The same source can have two valid programs with different masks/operators.
The paired pilot selects one answer per task to avoid confounding the supervision
comparison with repeated prompts. For the large mixture, compare all accepted
answers with one quality-ranked answer per task at equal tokens; do not assume
that a larger count of equivalent prompt/answer alternatives increases coverage.

RFT/TI rows can be added to count training supervision, even on the same task.
Also report unique task/source coverage. Remove identical supervision when
freezing a training snapshot; sampling weights do not create new production rows.

## Data-quality gates

1. Complete visible output, valid schema and runnable `edit(image)` program.
2. Source-only dependency audit: no target read, encoded reference bitmap,
   learned backend or external image. Archive the original teacher response.
3. Actual RGB execution and dimensions/dtype checks; reject unintended no-ops.
4. Image-level instruction/full-extent/preservation/quality checks. Existing
   `full_visible_target_v1` is our VLM judge, not official benchmark evaluation.
5. Source-grounded student explanations. The current paired build excludes
   references to an unavailable target/reference/second image. This text check
   does not prove that all target-informed reasoning is valid; inspect examples.
6. Native Qwen processor budget audit: full prefix/assistant boundaries, masked
   prompt tokens, no skipped overlong rows. Current pilot output ceiling is6144.
   The first4k-output audit rejected3264 new inventory rows;196 matched pairs
   remained. That4k snapshot is preserved but not the planned training treatment.
   Increasing the output ceiling alone is insufficient: audit the whole8k
   context with the actual image processor before training.
7. Freeze source-group and actual-RGB separation from development/test examples.

The existing judge accepts visual-quality score3 in some cases. Keep the original
judgments reproducible. Compare a separate quality4/preservation4 snapshot after
blinded auditing; do not silently reinterpret old pass labels. Scientifically
incorrect diagram text/values fail even if an image judge likes the appearance.

## Executable SFT comparisons

`scripts/build_paired_method_sft.py` pairs accepted English RFT/TI answers from
the same source task and teacher, verifies identical system/instruction and
actual source RGB, excludes evaluation groups/pixels, and freezes file hashes.
The aligned6k snapshot is `data/paired_method_sft_20261125/pairs_256_fb78a9e84ef0`. It contains256 tasks/250 source groups; all256 pairs share teacher/source/input.1455 inventory answers over6k were excluded before matching. Each treatment covers256 tasks with one answer per task:

| Treatment | RFT rows | TI rows | Assistant-token training budget |
|---|---:|---:|---:|
| ti00 |256|0|1,000,000|
| ti25 |192|64|1,000,000|
| ti50 |128|128|1,000,000|

TI percentage here is a row fraction. Token lengths differ, so count actual
per-method exposure from sampled IDs and precomputed token lengths; report
overshoot at the final optimizer step. All configurations use the same base
Qwen3.8-27B, rank16 standard LoRA, lr2e-5, seed20261125, accumulation2, 8192
context and262144 processor pixels. Python edits the native source image.
Use a second seed before drawing method conclusions.

This experiment is conditional on both methods succeeding. It isolates the
supervision alternative on matched tasks, not TI's rescue/coverage benefit.
Follow with RFT-only versus RFT+rescue at fixed tokens and matched source-family
sampling; disclose the changed task distribution. Then compare1k/4k/full data,
quality filtering, and answer-code versus evidence/plan/code supervision.

Do not fix a final universal 1:1 ratio before validation. Start from source and
operation coverage, monitor sampled token proportions, and cap repeated easy
recolors. Prioritize underrepresented TI, scientific figures and structural
operations whose teacher outputs pass the stricter quality checks.

## RL evidence and next check

The current one-shot smoke uses32 English training tasks and four independent
completions/group. Three genuine policy updates complete; initial/final reference
adapter SHA256 match. Equal-reward groups are skipped. Syntax-only execution
gets no success label; actual output images provide the reward.

On the same64 MagicBrush dev tasks the saved pre/post results are3/64 versus
11/64 VLM passes, with44/64 versus46/64 successful executions. Ten cases improve
and two regress. This is one seed under legacy greedy settings; known Qwen
numerical repeatability issues require new deterministic pre/post evaluations.
The second smoke uses seed126 and deterministic numeric settings. A genuine
update proves the engineering loop; reproducible held-out gains establish
learning. Inspect rewarded failures and all regressions for judge exploitation.

## Unattended work

Weekdays18:00–09:00 Beijing time and weekends use higher teacher caps. Independent
TI:61-Sol16,62-Sol16,GPT5.5eight,Terra/Luna4 each. RFT/rescue lanes have their own
caps:61 RFT16/rescueTI8;62 RFT8/rescueTI4. Caps are not throughput measurements.
Monitor actual in-flight requests, queue waits, accepted rows/hour and errors.
Retry transport/queue failures without relabeling them semantic failures.

A supervised supplementary TI producer has now been launched at
`data/editing_growth_v1/ti_night_sol_v2` (supervisorPID3972516). It covers3746
English training tasks:1003 MagicBrush,283 SciForma,1146 MaskEdit scene and1314
MaskEdit English. It uses a new `night_sol_v2` candidate namespace. Each candidate
is one independent API response; no prior failed program is supplied as history.
It schedules zero new requests during working hours and up to16 each for61-Sol
and62-Sol off-peak. Existing queued/in-flight API calls are never cancelled at
the morning transition. Across the currently configured producers, the off-peak
Sol caps are61=56 and62=44. These are request slots including queued work, not
measured throughput or a guarantee that the provider has no queue.

Its initial live refresh verifies NAS source/target files before the first
heartbeat; the routing-only dry run confirms the3746-task cohort without API
calls. NAS lookup latency must not be reported as model queueing. Launch/status
evidence is recorded under `outputs/english_teacher_supervision`.

## Construction audit and next snapshot policy

At14:48 Beijing time the new growth exports contain7333 accepted English
rows/5086 task IDs/4692 source groups; the historical+new inventory contains
19334 rows (13711 RFT+5623 TI). These are time-stamped inventory counts, not a
new model's effective training exposures. The growth-only audit is
`outputs/editing_construction_audit_20261125.json`.

The audit finds65 legacy candidate-ID collisions between independent rescue
and enrichment outputs, and zero completely identical supervision under its
task/source/system/instruction/answer key. Additional distinct answers can be
valid supervision. The next snapshot namespaces IDs by export origin and
answer hash, preserves original candidate IDs, deduplicates identical answers,
and refuses to overwrite frozen train/val payloads. Current live SFT datasets
remain immutable. All old provenance and weak-quality judgments remain visible.

There are492 accepted RFT/TI pairs sharing task/teacher/source-group/system/
instruction and matching no-op L1 metrics;436 have lower target L1 forTI. This
success-conditioned metadata/metric diagnostic does not reverify pixel identity,
establish semantic superiority or measure TI rescue. Use the actual-RGB-audited
paired training experiment for the method comparison. Its ti00/ti25/ti50 runs
finish157/153/150 optimizer steps with1001372/1001002/1003372 assistant tokens,
respectively, and are evaluating64 tasks under matching deterministic settings.

The next data recipe is organized around edit capability, with construction
mode and teacher as separate axes:

1. Local appearance edits: recolor, lighting, texture and retouch. Include
   full-object masks, thin boundaries and small/multiple objects; keep no-op,
   missed-target and spill controls for evaluation.
2. Structural edits: removal, addition and replacement. Retain teacher failures
   and quality scores to expose the classical-program feasibility ceiling;
   passing execution or a cartoon-like approximation does not establish success.
3. Scientific/text/layout edits: figure labels, legend/curve styles, correct
   symbols and readable text, spatial alignment. Check exact requested content
   and unchanged numerical meaning in addition to appearance.

Store edit categories from verified dataset annotation or audited labels.
Ambiguous/multi-operation instructions remain separately labeled. Do not
invent category coverage from dataset names alone. Sample by source/operation
and count actual assistant-token exposure. Initially cap two distinct accepted
answers per task (oneRFT and oneTI) for the primary corpus; compare unrestricted
accepted alternatives at the same tokens as an ablation. This is a proposed
snapshot policy, not a retroactive cap on ongoing production.

Build separate strict-quality candidates requiring visual-quality>=4 and
preservation>=4 along with current full-extent semantic gates. Preserve the
original pass labels; validate the stricter selection with blinded real images
before using it as the training default. Do not require positive global L1 gain
as a universal gate: legitimate alternatives and global edits can differ from
the dataset target. Evaluate completeness and preservation jointly to avoid
rewarding a mask that simply shrinks.

For the next scale experiment, compare nested1k/4k/full eligible task sets with
source separation, one-shot native serializer and explicit assistant/full-context
budgets. Keep a fixed-token experiment for data-size effects and a fixed-exposure
experiment for compute scaling. Do not choose a final TI percentage until the
current matched results and a second seed are available.

Actual token exposure is verified from sampled candidate IDs:TI token fractions
are0%,28.59%,53.71% for the0/25/50% row treatments. Mean visits per data row are
1.227/1.195/1.172. These are short pilots, not convergence evidence. Before
attributing a weak result to data scale, test longer matched4M/8M token budgets
on the same frozen corpus and record development performance, then enlarge
task coverage. Keep compute scaling distinct from changing the supervision mix.

The live dashboard publisher previously rehashed5562 accumulated thumbnails
(about308MB) per refresh. It now uploads the complete current page/rootJSON and
only its306 currently referenced assets, preserving old assets remotely. The
publisher was replaced after verifying its exact process identity; teacher and
GPU processes were untouched. Public visibility of each new refresh must be
checked independently from local rendering.

The primary9584 proxy was found absent and is now health-verified; teacher
results resumed. Its start recipe enables30-second queue retry and600-second
upstream read timeout. A separate9588 serves62-Sol. Shared quota errors also
occurred: maintain bounded/archived duplicate proxy logs and verify writes rather
than attributing connection errors to model queueing.

The ingestion reservoir previously paused at3GiB of images. The idle ingester
has been resumed under an8GiB cap, excluding Chinese infographic shards and
non-English instructions before new writes. Existing images/data stay frozen.
The current source materialization resizes to512x512 and records original
dimensions; it is not a claim of original-dataset native resolution. A future
scientific-figure branch must preserve aspect ratio and jointly test higher
teacher/student image budgets, text fidelity and whole-context limits before
merging it with this controlled pilot.

## Publication evidence still required

- A clear experimental claim: target-informed, executable pixel supervision
  improves source-only editing coverage/quality, with mechanism ablations.
- Large locked official evaluations, separated from the heavily reused dev64;
  label custom UltraEdit/SciForma holdouts accurately.
- Official benchmark metrics with no-op/source controls, plus calibrated VLM
  judging and a blinded human audit. Report coverage/preservation separately.
- Relevant visual-program and instruction-editor baselines at matched inputs,
  resolution and inference budget, with no target-informed oracle called a
  deployable baseline.
- Multiple seeds and scaling curves; paired RFT/TI, rescue/enrichment, reasoning
  format, quality selection, image/RGB inputs and SFT/RL comparisons.
- Released provenance/split/runtime manifests, visible supervision and code,
  actual execution outputs, complete API tokens/compute/latency and failure cases.

Official CVPR2027 dates were checked on2026-11-25: registration Nov10, submission
Nov16, supplementary Nov23 (all2026 AoE). Those deadlines have passed. Work can
support an already submitted paper; a new submission needs another cycle/venue.
Source: https://cvpr.thecvf.com/Conferences/2027/Dates
