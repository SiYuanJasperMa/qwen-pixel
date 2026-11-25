# Image-edit training recipe — 2026-11-25

## Objective and interface

Train Qwen3.8-27B to produce executable pixel-edit programs from a source image
and an English instruction. The program computes the RGB output with PIL,
NumPy and classical OpenCV. A target image is teacher-only construction
evidence; it never enters the student prompt or deployment inputs.

The agentic SFT correctness checks remain a separate diagnostic of Python,
image-observation and finish serialization. They do not establish one-shot
benchmark quality. Do not substitute a successful overfit or optimizer update
for held-out editing results.

## Construction

- Preserve source dataset, original task ID and split, image/source-group
  identity, instruction, teacher/model version, construction method, code,
  actual rendered output and verification results.
- RFT: source + instruction, one teacher response, actual execution and image
  judgment. No target image in generation.
- TI: source + instruction + target, one teacher response, source-only
  execution. Plans and visual evidence must be phrased for a student that sees
  only source + instruction. Do not supply a previous failed program as repair
  history.
- TI includes tasks already solved by RFT. This provides paired supervision
  and increases useful TI examples. Keep failed-task rescue tagged separately
  from independent TI enrichment.
- Multiple valid methods/teacher trajectories for one input can be SFT
  examples. Report sample rows as the primary production quantity. Separately
  report unique task/source coverage and remove completely identical
  supervision when freezing training snapshots.
- English policy: exclude MaskEdit CN, non-English language metadata, and CJK
  instructions/supervision. Retain scientific mathematical symbols such as Δ.
  Old datasets remain archived for reproducibility.
- Keep program, execution, non-no-op and full-target semantic filters. Analyze
  target consistency and teacher visual quality; do not relax filters to reach
  a numeric ratio. Inspect source/target conflicts before trusting TI labels.

## Active production configuration

At14:43 Beijing time, the shared English task pool contains14,298 tasks from MagicBrush, UltraEdit,
SciForma, AnyEdit, MaskEdit scene/English infographic and InstructPix2Pix.

Independent TI teachers: GPT-5.5, GPT-6.1-Sol, GPT-6.2-Sol, Terra and Luna.
Weekdays 18:00–09:00 Beijing time and weekends use the off-peak caps:

| Independent TI teacher | Off-peak cap | Working-hours cap |
|---|---:|---:|
| GPT-6.1-Sol | 16 | 4 |
| GPT-6.2-Sol | 16 | 4 |
| GPT-5.5 | 8 | 8 |
| Terra | 4 | 2 |
| Luna | 4 | 2 |

The existing RFT/rescue worker separately uses GPT-6.1 RFT 16 and rescue TI 8
off-peak. Actual execution/waiting counts are shown independently from caps.
Queue/API errors retry the same stage with backoff; retained outputs with
judge errors are judged again without repairing generated code.

A separate supervised `ti_night_sol_v2` line has3746 audited English training
tasks. It adds16TI request slots each for61-Sol and62-Sol only off-peak, with
zero new day requests. It waits for18:00 Beijing on weekdays and preserves
existing calls across the09:00 transition. The current combined night caps
are61-Sol56 and62-Sol44. Capacity includes waiting API requests; record actual
completed/accepted rows per hour separately. Initial live heartbeat confirms
`waiting_for_off_peak`,3746 tasks and zero in-flight requests during the day.

Artifacts:

- `data/editing_growth_v1/sft`: current English RFT/rescue export.
- `data/editing_growth_v1/ti_balanced_en_v1`: independent TI records and export.
- `data/editing_growth_v1/ti_night_sol_v2`: additional off-peak SolTI candidates.
- `data/editing_growth_v1/snapshots`: immutable English training candidates.
- `outputs/english_teacher_supervision`: live worker/publisher logs and PIDs.

Next frozen snapshots namespace candidate IDs by export origin and canonical
answer hash, retaining original ID/provenance. The growth audit found65 legacy
ID collisions across independent generated answers; differing answers remain
valid samples. Snapshot paths bind exact train/val payload content, and existing
frozen files cannot be overwritten. These changes apply to future snapshots;
live training inputs are not edited.

## SFT comparisons

Treat 1:1 RFT:TI as an experimental sampling target, not an established optimum.
Run 0%, 25%, 50% TI; consider TI-only after the mixed baseline works. Prefer
matched task, source pixels, instruction and teacher when available. Record
remaining teacher/data differences explicitly rather than attributing every
gain to TI.

Hold model, LoRA scope, learning rate, seed and total assistant-token budget
fixed. Track actual per-method token exposure and source diversity. Compare
epochs/data size only after serialization, masks and adapter reload pass.
Do not increase reported production size by duplicating rows to implement a
sampling weight.

Evaluate native RGB validity, code execution, full-target instruction success,
preservation and target similarity separately. Keep no-op and zero-shot
controls. Development examples guide changes; reserve untouched test/source
groups for final results.

## RL smoke requirements

The frozen initial SFT adapter is the reference. Score exactly the sampled
tokens with the same sampling temperature. Do not execute budget-truncated
outputs or fabricate reward variance. Use full-visible-target judging and
check reference weight hashes across updates.

Use multiple English training tasks and real independent samples. Require
finite nonzero gradients, changed policy weights and unchanged reference
weights. Evaluate the initial and updated adapters on the same held-out
instructions with the same decoding/runtime/judge. An optimizer update alone
is an engineering result, not evidence of a performance gain.
