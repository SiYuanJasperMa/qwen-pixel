# Qwen Pixel SFT correctness checks

The active objective is SFT correctness and stable independent execution. RL and bulk teacher collection are deferred.

## Evidence collected

- Training and inference use identical initial system, instruction, image pixels, and tool schemas on the exact replay task.
- Structured teacher function calls and the model's generated function calls serialize to the same Qwen conversation text.
- Manual shifted assistant-token cross entropy matches the model-reported loss.
- Cached and uncached generation agree on the audited training prefixes.
- A genuine native-photo trajectory, `coco_agent_008`, overfits after 120 steps: all four next actions match, an independent rollout completes `view_image -> python -> view_image -> finish`, and its RGB output exactly matches the real teacher result. The edit judge passes it.

These checks prove the single-trajectory training path and adapter reload. They do not establish held-out generalization.

## Actual problems identified

The clean-mask v2 checkpoint 400 completed 8/8 image-disjoint development rollouts but passed only 1/8 strict semantic checks. Format and execution success are insufficient. Old weak linear brightening could produce a near no-op, and some annotated instance polygons omitted visible target parts or included occluding objects. Reviewing 64 actual rasterized masks accepted only 40. These are training-label errors; training was stopped at step 765 instead of fitting them further.

The clean-mask v3 dataset rebuilds the accepted masks, uses gamma 0.75 for brightening and a 25% tint blend, and re-executes all tools so output images and feedback match the new programs. Every candidate reference is visually reviewed for full-target coverage and unrelated-region preservation before its complete trajectory is retained for training.

Greedy generation previously stopped at 1200 tokens and put truncated programs into subsequent history. It now retries the same input with larger budgets through 4096 tokens within the 8192 context, and never executes or appends a partial call. Evaluation caches now bind checkpoint, input, runtime, and decoding identities; completed image outputs can be rejudged without actor regeneration. Strict `full_visible_target_v1` rejects face-only edits for whole-person instructions. The final set is twenty frozen, evaluation-only official COCO val2017 photos with custom editing instructions, not an official image-editing benchmark.

Earlier low-loss claims did not separate actions. On an audited larger checkpoint, the Python edit's loss was about 1.38 while view/finish losses were about 0.0007–0.0009. Correctly generating a short tool invocation did not mean the model learned the edit program. The trainer now logs each sampled action and its loss.

Previous checkpoint loading used the original processor even when a trained processor was saved. New loading prefers the checkpoint processor. The newly saved agent contract also checks system text, tool schemas, tool-image role format, and vision token budget at inference. Mixed contracts are rejected during SFT preparation.

Historical images must be immutable. Existing snapshot machinery preserves actual source/crop/output observations. Live overwritten files cannot substitute for what the teacher saw at a past step.

## Minimal proven recipe

The current single-task check uses Qwen3.8-27B, LoRA rank 16 over standard attention and MLP projections, learning rate 2e-4, 120 optimizer steps, and one example per step. There are four next-action examples from one actual complete teacher trajectory. The vision pixel budget is 262144. Image editing uses the native RGB dimensions independently of the model's vision preprocessing. Older 9B runs are historical evidence; the deleted 9B checkpoint is not a current dependency.

Run `scripts/sft_27b_correctness_replay.sh single <fresh-output-directory>` on an unused authorized GPU from the shared project directory. It trains the adapter, checks all next actions after reload, and executes the task independently. This is an overfit diagnostic, not a training recipe for a large dataset. Set `CUDA_VISIBLE_DEVICES` to the chosen available GPU.

The next check, `scripts/sft_27b_correctness_replay.sh three <fresh-output-directory>`, uses three distinct real editing trajectories: vase recolor, monitor brightening, and donut blur. It trains 10 assistant actions, checks all teacher-prefix predictions, runs all three training tasks independently, and evaluates three separate photos. Both checks restore the taught checkpoint interface explicitly and retain strict semantic judging. The held-out subset is exploratory and does not constitute an official benchmark or a statistical claim.

## Completion criteria

1. Prompt/image parity and assistant-only labels pass.
2. A reloaded model reproduces all actions and actual pixels in a single overfit trajectory.
3. Several distinct training trajectories produce executable independent edits without repeated tool calls.
4. Held-out execution and semantic edit quality are measured separately. If these regress, do not call the broader SFT model ready.

A lower learning rate alone has already failed to resolve previous looping. Recurrent-gate LoRA changes, action mix, training exposure, and supervision diversity must be tested against actual action predictions and generated images rather than aggregate loss alone.

## Tool-call boundary recheck — 2026-11-25

The runtime parser previously accepted a missing `</tool_call>`, selected the
first action from two emitted calls, and silently replaced an earlier parameter
with a duplicate. These malformed generations must not count as successful
action replay. The parser now requires one complete native call, one function,
and the required declared parameters without duplicates or extra content. It
removes only Qwen's delimiter newlines so actual Python indentation and blank
lines survive round-trip serialization.

The targeted protocol, training-contract, coordinate, integer-coordinate, and
evaluation-manifest suite passed 26 tests and 6 subtests. A separate check used
the actual local Qwen3.8-27B tokenizer on the existing one-task and three-task
teacher datasets: all 14 action arguments round-tripped exactly and every
training assistant prefix matched. Results are saved in
`outputs/agent_parser_native_replay_20261125.json`.

This is a serialization check, not a fresh model-generated prefix replay or a
held-out editing result. Reloaded-model overfit, independent tool sequences, and
held-out RGB quality still need current-state verification. The SFT goal remains
active; no new RL or bulk collection run was launched for this check.

## Fresh Qwen3.8-27B checks — 2026-11-25

The 120-step single-trajectory run completed on the authorized f80f H20 worker.
Peak allocated memory was 60.60 GiB. After checkpoint reload, all four complete
tool calls and arguments matched their actual teacher prefixes. Independent
execution completed `view_image -> python -> view_image -> finish`, passed the
strict full-target judge, and reproduced the teacher's native 640x480 RGB output
exactly (mean absolute RGB error 0; PNG matches the submitted uint8 array).
See `outputs/agent_sft_en_27b_overfit_20261125/gate_summary.json`.

The first independent execution correctly rejected an interface mismatch:
the historical samples taught PIL/NumPy descriptions, while the current RGB
profile additionally describes cv2 and classical helpers. The explicit
`--checkpoint-interface` option restores the saved system/tool text only when
tool argument schemas and RGB runtime capability agree. User-prompt, image-mode
and pixel-budget validation still runs; mismatches are not bypassed.

Actual model forward-loss/cache audits are in progress. The first two actions
have identical reported and manually shifted assistant losses and matching
28-token cached/uncached greedy generations. The Python prefix has a nonzero
full-vs-prompt logit difference; its cause remains to be checked rather than
assumed benign.

The next fresh run uses Qwen3.8-27B and all ten actions from three real photos,
300 steps, rank16 standard LoRA and lr1e-4. Its attached pipeline checks every
teacher prefix, runs all training photos independently and evaluates disjoint
photos with strict judging. Artifacts:
`outputs/agent_sft_en_27b_three_tasks_20261125`. No multi-photo or held-out
success is claimed before those results exist.

The controlled causality probe is now complete on the Python prefix: replacing
future non-special target tokens at the same sequence length changed the
pre-action logits by exactly zero; a repeat forward was also identical. Changing
the sequence length produced the previously observed numerical difference.
This establishes no detected future-token influence for the audited prefix;
it is not a universal proof across every model/input.

The actual independent Python prefix also matches the teacher prefix in every
encoded tensor: input IDs, attention mask, multimodal type IDs, pixel values and
image grids. Both raw generated assistant text and reconstructed structured
calls match. See `actual_prefix_tensor_parity.json` in the single-task run.

Checkpoint restoration now rejects checked/unchecked RGB feedback mismatches
even when argument schemas agree. SFT preparation also rejects non-assistant
targets and multi-call targets for the single-action runtime. A targeted numeric
test encountered one process timeout under concurrent load and succeeded on
isolated reproduction; Python math/OpenCV subprocess threads are now bounded.
The latest targeted suite passed 25 tests and 6 subtests. Multi-photo training
and held-out verification remain in progress.

All 14 actual examples from the one-photo/three-photo pilot datasets now pass
the complete assistant-only label audit: every historical token is masked,
every next-action token is supervised, and all prompt/full multimodal tensors
match across the boundary. The artifact is `all_pilot_loss_masks.json` in the
single-photo run directory.

Adapter loading now validates the declared training base before loading GPU
weights, also checking saved training arguments where available. Canonical
symlinks to the same base are allowed; different same-named directories and
conflicting training metadata are rejected. Thirteen focused identity,
contract and image-observation tests passed. These identity checks do not claim
a cryptographic audit of all original base-weight bytes.

New contracts persist structured execution features: RGB globals, crop support,
numeric rejection, per-call Python scope and persistent task files. The runtime
passes its actual profile flags for validation. This covers clean-edit tool
descriptions that say `validates` rather than `rejects`, and prevents changed
execution behavior from being hidden behind matching prompt/schema text.
Legacy contract signatures remain unchanged when the feature field is absent;
restoring a legacy checkpoint still compares its inferred behavior. Thirty-one
targeted tests and six subtests passed, and both fresh single/three-photo legacy
contracts were checked successfully after this change.

The fresh three-photo run completed all 300 updates. Reloaded teacher-prefix
generation now passes 10/10 complete calls and exact argument matches. Training
photo and held-out autonomous RGB results are still being collected.

The updated visual-conditioning diagnostic uses the checkpoint's declared base
and selects real Python-action rows without relying on one historical ID format.
For each source it substitutes another image at identical image/token geometry,
without writing training examples. The three measured Python actions increased
assistant-token loss by 0.174, 0.220 and 0.185 nats (mean 0.193). This shows a
measured response to changed visual content; it does not prove correct object
localization or held-out editing. These legacy programs have no tagged geometry
tokens, so geometry loss is reported as unmeasured rather than NaN or zero.

## 37-photo RGB diagnosis and matched continuation — 2026-11-25

Both matched 400-step Qwen3.8-27B adapters are now saved. The attention model's
first twelve teacher-prefix predictions contain twelve complete calls but only
eight exact argument matches. Executing the three predicted Python actions under
the same verified teacher history produces valid native RGB in all three cases.
Their changed-footprint IoUs against re-executed teacher programs are 0.608,
0.830 and 0.686; none reproduces the reference RGB exactly. These footprint
statistics describe actual changed pixels, not annotation-mask or semantic
accuracy. The principal visible differences are polygon coordinates rather than
the common RGB-computation boilerplate.

The independent eight-photo development evaluation is ongoing. Its first three
attention-model tasks complete the taught four-action sequence. Full-target
semantic judging accepts person blur, rejects book brightening for spill into
neighboring content, and rejects broccoli tint for incomplete coverage and an
unnatural geometric overlay. The accepted person result is independently checked
as uint8 RGB at the source's 640x480 dimensions, with identical PNG/NPY pixels and
zero parse/Python errors. These three observations are preliminary diagnostics,
not a completed eight-task success rate or an official benchmark result.

`scripts/audit_generated_prefix_pixels.py` now isolates next-action localization
and RGB failures with freshly re-executed reference/prediction histories. It
refuses partial sampled calls. One history replay in the twelve-row CPU audit
timed out at the unchanged 30-second limit; the isolated fresh replay succeeds.
This is recorded as a diagnostic timeout, not a model editing failure. Five
focused tests cover actual RGB execution, no-op/over-edit distinctions, partial
call rejection, repeated identical calls and pending semantic judgments.

Two matched continuations start from the same attention checkpoint and data:
400 additional steps, lr5e-5, rank16, seed42 and identical image/runtime settings.
Only the coordinate loss weight differs (1 versus 4). Both have confirmed
nonzero gradients and adapter updates. The actual first region/Python tensors and
labels remain unchanged with weight4; historical tokens retain zero weight.
The frozen comparison is
`outputs/agent_sft_en_27b_geometry_continuation_20261125_comparison.json`.
Their pipelines evaluate independent development images before prefix replay.
No improvement is claimed until that evaluation exists.

Two separate GPUs generated different polygon strings from byte-identical
teacher rows under greedy decoding. A completed probe on one reloaded model also
finds a difference on an immediate repeat at output token130 and after another
image at token128. All encoded input fingerprints match. Two uncached forward
passes performed after the generations have identical logits. No uncached
forward occurs between the generation calls under test. These 256-token outputs
are diagnostic fragments and are not executed or counted as complete actions.
The cause remains unproven: the next comparison enables deterministic Torch,
cuDNN and cuBLAS settings without modifying the actor/runtime of existing
evaluations. See `generation_repeatability.json` in the attention run directory.

Latest machine-readable training/evaluation state:
`outputs/agent_sft_en_27b_status_20261125.json`. The objective remains active:
regional preservation, full-target coverage and complete development reliability
are not yet established.

### Deterministic decoding and recurrent state precision

The deterministic comparison has completed: all three same-image 256-token
generations match, including a repeat after another image. It enables
`torch.use_deterministic_algorithms(True)`, deterministic cuDNN, disabled TF32,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The model uses FLA chunk/recurrent GDN
kernels. This proves repeatability for the tested prefix length and does not yet
prove complete-action repeatability or editing quality.

A separate cache discrepancy is now reproduced in regression tests. Installed
`LinearAttentionLayer` initializes its recurrent-state buffer using the
convolution cache dtype. With bfloat16 convolution and float32 returned recurrent
state, float32 values such as1.001 are rounded on storage. The local
`pixel/generation_cache.py` retains the returned recurrent-state dtype while
keeping convolution state and full-attention layers intact. Two regression tests
pass, including fresh independent caches, in-place updates and reset. The current
GPU precision probe has not yet established complete-generation or held-out
quality improvements; existing formal development actors remain unchanged.

## Decoder comparison and exposure audit

The original 400-step development evaluation completes8/8 native RGB edits for
both models. Standard passes2/8 strict semantic checks; attention passes3/8.
With unchanged weights/images, the standard decoder comparison completes8/8 in
each condition and passes4/8 with deterministic decoding versus1/8 with retained
float32 recurrent state. Higher cache precision is not automatically a quality
improvement. Deterministic decoding is the current long-training choice; the
full attention-model decoder comparison is still pending.

The192-token teacher-forced numerical audit measures small KL divergences from
full-forward logits (0.000272 for installed cache,0.000338 for float32 state).
The92 geometry tokens still have approximately0.956 target NLL. This one-prefix
audit does not establish universal cache equivalence; it shows that cache
precision is insufficient to explain this sample's high geometry loss.

All236 verified training histories are unique under canonical visible text,
RGB content/dimensions, tool schemas and image mode. Their177 observations pass
hash verification again. No divergent labels for an identical visible history
are found. See `visible_supervision_audit.json` in the frozen data directory.

Actual sample-count logs reveal an exposure mismatch: both successful tiny
pilots train each action30 times on average, while the37-photo400-step runs
average only3.39 visits/action. Their completed800-step continuations would still
average6.78. With236 rows and accumulation2,3540 total steps match the pilot's
mean30 visits/action. `scripts/sft_37photos_exposure30.sh` adds2740 steps after
the saved matched800-step weights, with native development edits every400 steps.
Higher exposure is a learning hypothesis, not a promised semantic success.
New training logs also record actual sample IDs so exposure can be audited
directly instead of inferred only from action counts.

## Live-optimizer validation smoke

The optional inline editing validator preserves module modes, Python/NumPy/
Torch RNG and the cache setting; clears ephemeral multimodal position state;
and checks that trainable weights do not change during validation. It protects
lazy setup and imports and uses fresh workspaces when old artifacts exist.
Validation outputs do not enter the frozen training JSONL.

A real A800 Qwen3.8-27B smoke continues the existing120-step adapter for20 updates.
At steps10 and20, independent edits pass strict judging and exactly reproduce the
immutable teacher RGB (MAE0). Training resumes after step10 with nonzero gradients
through step20. Reloaded prediction matches all4 complete teacher actions and
arguments. Artifacts:
`outputs/agent_sft_en_27b_inline_validation_smoke_20261125`.
The current state/cache policy tests pass8 cases; fresh-workspace/state tests
pass3 after the additional artifact-isolation change.

The frozen20-image final set is
`data/independent_coco_edit_eval_v2/tasks.jsonl`. Its image hashes, bounding boxes
and evaluation-only flags are checked. Scene IDs and decoded RGB have zero
overlap with these37 training photos; final/development scene overlap is also0.
This checks post-training separation, not base-model pretraining exposure.
Do not call the objective complete before reliable actual held-out edits exist.

## Matched image-dependence check during exposure training

Both longer A800 runs have confirmed nonzero gradients, real adapter-weight
updates and identical sampled action IDs across the checked shared initial
steps. They use the saved800-step condition-specific adapters,2740 further
steps, the same data/settings and explicit deterministic decoding for validation.

The image-perturbation auditor now selects distinct source photographs rather
than distinct trajectories; different tasks on one photograph can no longer
serve as each other's image donor. It verifies original observation bytes before
constructing any intentional diagnostic substitution. The altered observations
are confined to diagnostic outputs and never enter training.

The same six teacher prefixes, donors and geometry-token spans are tested at
400 and800 total steps. Mean geometry NLL is0.691 at400,0.575 at800 with ordinary
loss, and0.572 at800 with geometry weight4. Swapping to another photograph raises
geometry NLL in all six cases: mean increases0.114,0.222 and0.215 respectively.
This demonstrates measured visual dependence and better teacher-prefix geometry
learning. It does not prove correct localization or independent edited-image
quality, and the two weighting conditions are close on this small diagnostic.
The comparison is
`outputs/agent_sft_en_27b_matched_visual_comparison_20261125.json`.

The short800-step independent development evaluations and the long-training
400-step native checkpoints are still pending. Regional coverage/preservation,
completed calls and loops must be assessed from actual generated trajectories.

## Actual alpha and non-target RGB diagnostics

The complete800-step development runs produce native RGB for8/8 tasks each;
ordinary weighting passes2/8 VLM checks and geometry weight4 passes4/8. These
VLM labels are not sufficient evidence of correct regional editing.

Actual emitted programs are re-executed in fresh isolated diagnostic directories,
using verified source pixels and the model's own region selection. The eight
weight4 RGB results reproduce the original submissions exactly. The alpha
variable is checked to contribute to the final emitted output; diagnostic support
is normalized by alpha's own peak so geometric coverage is separated from opacity.
COCO instances are resolved by exact image/category/native box, rejecting ambiguous
instance matches. No instance masks are given to the actor or added to training.

Concrete failures: book ROI covers99.1% of the annotated object, but alpha covers
69.1% and47.1% of alpha mass lies outside the object. Broccoli's initial ROI covers
only46.0% of the target and final alpha covers21.2%. These distinguish grounding
errors from polygon/coverage errors. See `region_diagnostics/roi_vs_mask_coverage.json`
under the geometry4 short-continuation output.

Some VLM passes miss material preservation errors. The accepted left-giraffe
program also touches the overlapping front giraffe; after allowing a2-pixel
boundary band,7,188 non-target pixels change by more than1 RGB unit and23.0% of
total RGB delta energy lies outside the annotated target. Raw source, annotation,
predicted alpha and actual edit are rendered in that case's `comparison.jpg`.
These spatial measurements are diagnostics, not official image-editing scores.

`pixel/region_preservation.py` adds the same measurements to long-run inline
validation. The optional evaluation-only registry binds exact task bytes, source
file/RGB, target box and mask hashes. All8 original task images match the replayed
source RGB and the registered path has been exercised on all8 actual edits.
Six focused preservation/state tests pass. Actor instructions and training JSONL
are unchanged. Do not interpret a VLM pass as regional correctness when the
pixel evidence shows substantial leakage or missing coverage.

A four-photo, paired correct/partial-crop action probe is running separately.
It tests preferred re-localization behavior without adding any probe row to SFT.
Choosing another next action alone is not classified as a semantic edit failure.

## Crop-feedback recovery SFT experiment

The paired crop probe completes all8 native next-action generations. With a
correct crop the four photos choose Python; with deliberately incomplete crops
all four still choose Python instead of re-localizing. Replaying those four
programs produces valid RGB but changes zero pixels outside the partial ROI,
while verified teacher references change8,776,34,300,7,717 and6,242 target pixels
there. This execution evidence confirms omitted target editing for these probes;
the action-name mismatch alone was not used as semantic failure evidence.

A bounded paired curriculum uses these four existing training photographs,
keeping the main236-row datasets untouched. The correct-crop references retain
their original four actions. In the recovery histories, an actually executed
partial source crop is followed by the correct source crop, Python, edited-image
inspection and finish. The intentionally bad initial crop is history only and
is never a supervised target. All four recovered final RGB arrays exactly equal
their original verified references; no new teacher API is used.

One initial local reference replay timed out at the unchanged30-second limit and
was not accepted. A fresh worker build completed successfully. Its32 rows and
the32-row positive-only control both pass complete assistant-label and native
prompt/full tensor checks, with maximum paired sequence length3950. The control
uses exactly the same target actions/order and training budget, with duplicated
correct-crop history replacing incomplete-crop recovery history.

Both480-step experiments have live processes and confirmed first-step nonzero
gradients/adapter changes. They use the same800-step weight4 adapter, rank16,
lr2e-5, ordinary loss, accumulation2 and deterministic decoding. Independent
development editing runs every240 steps; after480, the original8-prefix crop
probe tests both retention on valid crops and recovery on incomplete crops.
This is an isolated feedback-learning experiment, not a replacement for the
main37-photo exposure runs or a completed reliability claim.
Settings: `outputs/agent_sft_crop_feedback_comparison_20261125.json`.


## Full crop continuations and exact repair references

The four-photo paired480-step model now executes eight complete continuations
from verified teacher observations in fresh actor workspaces. All8 produce
native RGB; all4 correct-crop controls exactly equal accepted reference RGB.
Incomplete-crop histories re-localize4/4 and recover exact RGB3/4; the remaining
cow edit has0.99584 reference-change recall and0.03132 native RGB MAE, with the
last two polygon vertices replaced by one point. This is a training-photo prefix
continuation diagnostic, not an initial-task or held-out quality score.

The positive-only480-step model also completes8/8. Correct crops retain exact
RGB4/4, but incomplete crops never re-localize and achieve reference-change
recall0.339/0.529/0.730/0.600. All8 original histories reproduce native input
tensors after fresh runtime restoration. The four paired model-generated
follow-up Python prefixes also equal their actual SFT prefixes across input
IDs, masks, image tensors, image grids and multimodal type IDs. The residual
cow geometry error is not explained by a changed visible prefix.

A numerical audit requesting the whole teacher target failed because the
native template adds a newline after assistant EOS. That newline was formerly
also a supervised label, though actor generation stops at EOS. Canonical SFT
now masks the conversation trailer, retaining EOS supervision and unchanged
prompt/image inputs. Exact sampled-token scoring is left intact, including
truncated samples. Real Qwen tokenizer/processor tests cover trailer masking,
stop markers inside code, and exact sampled token preservation; eleven relevant
label/geometry/replay/clip tests passed. The long cache audit is rerunning with
only the contiguous supervised assistant action and explicit per-token logits.

Semantic-repair construction now accepts frozen approved source/target
observations rather than mutable execution files. Padded source re-cropping
retains the original alpha clipping region, and every repaired RGB must equal
the frozen reference. Eight existing training photos produced16 valid repairs
with64 correct action labels; the synthetic wrong actions remain history only.
The original-crop variant keeps the original verified code and geometry. Its
matched comparison combines236 frozen normal actions with64 feedback/control
actions. Both300-row conditions have identical targets/order and final RGB.
Feedback training is starting on A800GPU2; control waits for the verified
existing exposure pipeline on GPU7. Both use the same2400-total-step initial
adapter, rank16 attention, lr2e-5, seed42,4500 additional updates (mean30 visits
per row), native development evaluation every450 updates.

Latest2400-total-step regional audits replay16/16 native submissions exactly.
The weighted giraffe case clean_development_04 passes the VLM but has0.301
target alpha support coverage and0.506 outside-target alpha mass. Annotated
source/overlay/actual RGB confirm a missed body and edits on the front giraffe.
Do not use VLM passes or low loss to mark reliable editing achieved.


The1346-token audit is complete for both caches. The installed bf16 state
cache differs from full forward at token996, within the cow polygon point
`[0.2721,0.6074]`; full-forward teacher-token NLL is0.01257, cached NLL0.84796
and local KL0.77858. All other argmax positions agree. Retaining fp32 recurrent
state gives1346/1346 argmax agreement, reducing mean KL0.001292→0.000369.
This explains a concrete cached-decoding geometry mismatch under an identical
verified source/prefix. It does not prove unseen mask quality. Eight full crop
continuations plus the fixed development8 are rerunning with fp32_state before
changing the default policy; earlier fp32 held-out scores were mixed.

New300-row corrective SFT has confirmed nonzero gradients/real first-step LoRA
updates (B delta L1=16.205), peak allocated memory about70.4GiB on A800GPU2.
Label policy is recorded as native_assistant_through_eos_v1. The matched control
remains queued behind the live exposure pipeline on A800GPU7.


A real20-step27B label-policy regression is now running from the proven120-step
adapter on A800GPU0, with native editing checks at10/20 and four-prefix replay
after reload. First-step LoRA B delta L1=20.893 is nonzero. The precision cache
continuation currently has3 completed cases, all exact native RGB; all8 and independent
development results remain pending. Source/task and checkpoint match the earlier
regression; no new toy task or teacher data is substituted.

The new EOS-label regression checkpoint10 completes the native four-stage
editing task and passes the unchanged VLM gate; independently loaded native
RGB equals the frozen reference at640x480x3 withMAE0. Mode/RNG restoration
and unchanged validation weights are confirmed. Checkpoint20 reload/prefix
checks remain pending. The precision-cache full-continuation run now has4/4
completed exact-RGB results; all8 plus independent development are still required.


## Completed label-policy regression and precision recovery

The20-step EOS-label regression now reloads and reproduces all4 complete
native target actions exactly. Both checkpoint10 and20 execute the full task
and reproduce frozen reference RGB withMAE0.

The float32-state paired480-step model completes all8 correct/partial-crop
continuations with8/8 exact native RGB, fixing the cow tail coordinates while
retaining all4 correct-crop references. These remain supplied-prefix tests on
existing training photographs. Independent development8 is still running.
Its first two measured outputs preserve the person reasonably but still fail
the second object's targeting; do not infer unseen editing reliability from
the8/8 exact training-photo recoveries.

The300-row feedback dataset has300 distinct verified visible histories; its
control has236 distinct histories and32 duplicate groups. Both have zero
divergent next-action supervision and verified observation snapshots. Inline
validation now fingerprints the loaded decoder/cache factory/cache initializer
as well as the loaded rollout, preventing changed numerical code from reusing
a prior evaluation under the same policy name. Seven implementation-identity,
manifest and validation-state tests passed.


## Independent baseline and validation import regression

The paired480 adapter's completed fp32 independent development8 produces8/8
native RGB and3/8 custom VLM passes. The untuned Qwen3.8-27B with the same
interface, source tasks and precision policy produces7/8 native completions
and1/8 passes. These do not establish reliable whole-object editing. Source
crop coverage for book and broccoli is only0.124 and0.462; other crops mostly
cover the object but their predicted masks/edits miss or touch unrelated areas.

Adding a decoder convenience export while a training process held an older
module caused corrective SFT validation setup to ImportError at step450. No
model editing validation ran at that point. The saved450 adapter is preserved;
its optimizer was not saved, so resuming it cannot claim exact optimizer resume.
Validation now hashes existing loaded decoder/cache objects directly, and the
trainer binds its validator before optimization. Five state/compatibility/loaded
identity tests pass, including deletion of the newer convenience export from
the cached module. A fresh reproduction from the same initial adapter is live:
its first71 sample/loss records exactly reproduce the preserved run. The saved
450 adapter is separately being evaluated before deciding further curriculum.

Twenty-four actual incomplete/displaced crop histories on12 already-approved
training photos produce96 positive correction actions. All24 final RGB arrays
exactly match their original immutable references; erroneous crop actions are
history only. Their actual observed boxes equal the reported omission boxes.
A396-row crop/control candidate holds the300-row base and added target actions
identical. It is prepared data only; no third training experiment is launched.


## Exact training continuation

The trainer now saves a complete local continuation state bound to the exact
adapter-weight SHA, data SHA, row IDs, label policy and numeric/training
settings. It includes AdamW state, sampler permutation/cursor, supervised-token
count and Python/NumPy/Torch/CUDA RNG. The state is written atomically before
native validation. Default storage keeps only the latest state; the regression
option retains checkpoint states. An explicit stop-after-step does not alter
the planned optimization horizon. Changed training identities/checkpoint weights
are rejected, and nonempty output directories require explicit resume state.

A real27B comparison is now complete: uninterrupted12 updates versus stopping
at4 and restoring all training state produce identical sample IDs, losses,
learning rates and supervised-token counts for all12 steps, identical final
LoRA tensors, and exact native edited RGB at8/12 matching the frozen accepted
reference. A CPU AdamW interruption/restore comparison also passes. This closes
the new continuation mechanism regression; it does not retrospectively recover
the optimizer missing from the earlier failed450-step run. That run is being
reproduced from its original initialization, with its prior sampled/loss prefix
checked exactly, while the preserved450 adapter remains separately evaluated.

The preserved450 adapter completes8/8 native edits and4/8 VLM passes under
fp32-state decoding. The parent-adapter matched evaluation is still pending;
this is not yet an attributed feedback-data improvement or reliable editing.


## Whole-target versus preservation after feedback450

The initial2400-step adapter and450-step feedback adapter are now both tested
on the same development8 with fp32-state decoding: both complete8/8; VLM passes
change2/8→4/8, gaining person00/person05 without losing a prior pass. There is
no matched450-step ordinary-control result yet, so extra training versus
feedback-history effects cannot be isolated.

Actual code/alpha replays reproduce all16 submitted RGB arrays. Gains involve
less spill but also smaller target support: person00 coverage0.886→0.714 and
outside alpha mass0.246→0.038; person05 coverage0.927→0.815 and outside mass
0.277→0.173. Book coverage falls0.072→0 and other failures remain. Do not
conflate the VLM pass increase with full-target reliability.

The synthetic partial rectangle mistakes also included spill. A separate
clipped-contour pure-omission construction now retains only part of the target:
eight actual bad images cover0.118–0.252 of the approved target mask and change
zero pixels outside its2px boundary band (RGB difference>1). All eight repairs
exactly reproduce frozen reference RGB; wrong actions are not supervised.
The32-row data is a pending candidate, not a trained-model improvement. Three
contour/clipping tests passed. These examples separate expanding a missed
target from removing an unrelated edit rather than rewarding mask shrinkage.

Reproduction reached450 with adapter-file SHA exactly equal to the preserved
failed run. The previously failing native validator now runs and saves cases.
Complete8-case validation and resumed optimizer progress remain to be checked.

## Verified continuation and new construction identity audit

Reproduction checkpoint450 native development validation is complete:8/8 RGB
completions,4/8 custom VLM passes. The saved validator reports trainable weights
unchanged and training modes/RNG restored. Subsequent optimizer updates are
present (step479 observed, finite gradient); the prior validation ImportError
is resolved. The matching ordinary-control trainer is genuinely live onGPU5
(trainerPID1337268, step53 observed); the short-lived launcherPID1337618 has
exited. Check the trainer or output records rather than treating a launcher exit
as training failure. No matched-control450 method comparison is available yet.

The English growth exports contain65 reused candidate IDs across independently
generated rescue/enrichment answers. Their supervision contents are distinct.
The future growth snapshot builder now derives IDs from export origin, original
ID and canonical answer hash, retains original provenance, and binds snapshot
directories to train/val payload hashes. Existing frozen files cannot be
replaced. This does not rewrite data being used by current training. Two tests
verify independent legacy-ID answers remain addressable and frozen payloads
cannot be overwritten; with teacher route tests,10 tests pass.

New broad one-shot production/RL exploration follows the user's renewed
authorization. Agentic recovery remains an independent correctness diagnostic,
not the one-shot editing benchmark or evidence of general image-edit quality.

## Actual held-out cache audit and complete sampled-token exports

Two actual visible Python actions from feedback450 fp32 development outputs
(person00 and broccoli02) have937/1093 canonical SFT target tokens. Their native
SFT/audit input IDs, attention masks, multimodal types, pixel tensors and image
grids exactly equal the fingerprints recorded by the original rollouts. The
original sampled token IDs were not saved in those older exports; matching
target counts does not prove original token-by-token identity. This audit forces
the canonical SFT tokens, not a reconstructed original sampling trace.

On person00, cached/full-forward argmax disagreements change11→17 when retaining
fp32 states, while meanKL decreases0.009542→0.006021. On broccoli02 disagreements
change28→20, with meanKL0.031840→0.013720. Thus the earlier training-photo1346/1346
agreement cannot be generalized to unseen histories. Native input parity is
proved for these cases; full/cache numerical equivalence is not. No global
decoder default was changed. Evidence:
`outputs/agent_sft_heldout_cache_parity_20261125/summary.json` and
`prefix_tensors.json`.

The same450 weights complete8/8 and pass4/8 under both deterministic and fp32
settings, but actual output regions differ. In both settings book/broccoli crops
cover only0.423/0.446 of the annotated targets. The right-hand man is largely
missed despite a crop covering0.999 of his target. A direct source/target-mask/
output contact sheet confirms a blur over his hand/bottle area rather than his
visible body. Region difference fractions are diagnostics, not semantic mask
coverage: a correctly blurred uniform surface can have few changed pixels.
The evaluation-only overlays are `outputs/agent_sft_feedback450_failure_regions_20261125.jpg`.

Rollout SFT export previously hardcoded`qwen_bus_rollout` for every photograph:
32 actual exported actions had only4 distinct IDs. New exports derive IDs from
the actual history/model/adapter; executing that identity function on those32
saved actions produces32 distinct IDs without rewriting old exports. The new
export also retains each complete action's exact sampled token IDs throughEOS,
verifies its observed prefix and visible text match the saved assistant turn,
and rejects truncated/mismatched actions before writing a training file. Canonical
teacher data remains supported. Export/nativeEOS unit tests pass5/5; the real27B
single-photo export/reference-RGB regression has been launched separately, with
no claim of unseen quality from this test.

The previously audited396-row combined mask/crop-history comparison is now
launched onA800GPU6/7 (wrappers1339648/1339649; trainers1339650/1339651). Both
start from the same2400-total-update parent and use identical target actions/
order,480 steps, accumulation2, rank16 attention,lr2e-5,seed42 and deterministic
native development checks at240/480. Both have reached32 actual updates with
finite nonzero gradients. New96 actions cover24 partial/displaced-crop histories
on12 existing training photographs; all references remain exact nativeRGB.
Both conditions share the same300-row semantic-mask-feedback base. Only the
additional96 actions differ in observed crop-recovery versus ordinary correct
history. This isolates added crop-history supervision conditional on that common
feedback base; it does not isolate mask-feedback effects. Existing300-row runs
and data are separate. Actual first300-row equality is verified in
`outputs/agent_sft_mask_crop_pair_20261125/design_audit.json`.
The new pair's first8 updates have identical sample token counts4492 and identical
learning rates; observed maximum training allocation is68.68GiB. Checkpoint
optimizer/sampler/RNG state retention is enabled for correct continuation.

The real Qwen export regression is now complete. All four native calls finish;
the saved target IDs equal the actual sampled IDs, re-encoding preserves all
prefix masks and assistant labels throughEOS, and native outputRGB is exactly
equal to the frozen checkpoint20 reference. No target/reference was read until
after actor completion. Evidence:
`outputs/agent_sft_native_export_regression_20261125/result/summary.json`.
This closes the new export's integration regression on an existing training
photo; it is not additional unseen-task success.
