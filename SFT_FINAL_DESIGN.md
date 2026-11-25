# Qwen Pixel-Edit: final SFT design and ablations

> 最新数据构建方案见 [DATA_CONSTRUCTION_PLAN_V2.md](DATA_CONSTRUCTION_PLAN_V2.md)。
> 本文保留旧实验设计与审计依据；旧来源比例不代表当前已选定的最终配方。

Updated: 2026-11-22. This is a proposed controlled recipe, not a claim that the final dataset or runs below already exist. Existing v12–v15 runs are exploratory results.

## 1. Separate dataset from construction mode

Every edit pair `(source, instruction, target)` can generate two kinds of supervision:

| Mode | Teacher generation input | Student input | Supervised output |
|---|---|---|---|
| RFT / source-only | Source image + original instruction | Source image + original instruction | Visible structured evidence, plan, executable code |
| TI / target-informed | Source image + instruction + target image | Source image + original instruction | Visible structured evidence, plan, executable code |

Both modes use one independent teacher response per candidate, followed by offline execution and verification. Repeated candidates are independent samples, not repair conversations. Neither mode calls a pretrained image generator. The code operates on the provided source RGB using PIL/NumPy/OpenCV.

TI+mask is a third experimental condition, not part of the default TI prompt. Target visibility is privileged information during synthesis; runtime code must not depend on target/mask files or literal dumps. Normal coordinates inferred from the source, including source-relative coordinates, are legitimate. Banning every coordinate would exclude normal image editing programs.

## 2. What related work actually says

**STaR (Zelikman et al., NeurIPS 2022)** first generates unhinted solutions, keeps correct ones, then rationalizes failures with the correct answer as a hint. Its training set is the union of unhinted correct solutions and successfully rationalized failures. There is no universal fixed mixture percentage: it depends on the pass/recovery rates. The answer hint is removed from the training input. Original STaR generates the next round with the improved model but fine-tunes again from the original model rather than repeatedly continuing one checkpoint.

- Paper and Algorithm 1: https://ar5iv.labs.arxiv.org/html/2203.14465
- Archival paper: https://arxiv.org/abs/2203.14465

**RFT (Yuan et al., 2023/ICLR 2024)** collects distinct correct model-generated reasoning paths, including paths from multiple models. Source-only filtered distillation is a useful baseline by itself, without answer-conditioned generation.

- https://arxiv.org/abs/2308.01825

**ReST-EM (Singh et al., TMLR 2024)** iterates generation, binary correctness filtering, and supervised fitting; it caps solutions per problem to prevent easy tasks dominating. Its paper specifically reports that answer-hinted rationalization increased false positives with correct answers but incorrect reasoning in preliminary experiments. Thus mixing TI is supported by STaR, but TI quality is an empirical question.

- https://arxiv.org/html/2312.06585v3

**MetaMath (Yu et al., ICLR 2024)** uses explicit construction quotas: its 395K corpus contains 155K answer-augmentation, 130K rephrased-question, 55K self-verification, and 55K FOBAR examples. The GSM8K component is 80K/80K/40K/40K (2:2:1:1). These are quotas for distinct augmentation/task types, not evidence for an optimal RFT/TI percentage. In particular, backward-task inputs can legitimately include a known answer as part of the newly defined inverse task; that differs from removing a privileged target hint during distillation.

- Table 1: https://arxiv.org/abs/2309.12284

### How to read mixture rules from these papers

STaR's two pools are combined after filtering rather than sampled at a fixed 75/25 ratio. With one retained candidate per task, source-only success fraction `p` and TI recovery fraction `q` among failures, the TI fraction of accepted examples is `(1-p)*q / (p+(1-p)*q)`. For example, 400 source-only successes plus 300 recovered failures out of 1,000 tasks yield a natural 4:3 mixture. This is a hypothetical explanation, not a measurement of our teachers.

ReST-EM instead trains on filtered generated outputs and does not augment the iteration's training corpus with human-written outputs in its reported recipe. Its per-problem cap of 10 addresses task imbalance. RFT merges and deduplicates distinct successful paths, including multi-model pools. MetaMath fixes augmentation counts and verifies the value of augmentation through comparisons. These rules operate on different axes and cannot be collapsed into one universal teacher-RFT/teacher-TI ratio. Our 75/25 remains only a controlled candidate; include STaR's natural successful-plus-rescued union as a production-policy baseline.

**Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation (2026 preprint)** reports controlled degradation from rationalize-toward-the-answer prompts in math/code. This does not establish the effect for image editing, but supports a matched-task comparison and careful inspection of target-dependent explanations rather than assuming TI is superior.

- https://arxiv.org/html/2607.14552

Our current setup is **teacher distillation with STaR-inspired rationalization**, not the complete STaR self-training algorithm. Continuing SFT on the same dataset does not constitute a new STaR round; a new round must regenerate and verify candidates with the updated model.

## 3. Audit of existing artifacts

Source: `outputs/sft_provenance_audit_20261122.json`, computed from actual JSONL rows.

| Dataset snapshot | MB source-only | MB TI | UltraEdit source-only | SciForma source-only | Total | TI row fraction |
|---|---:|---:|---:|---:|---:|---:|
| v12 | 5,250 | 532 | 1,996 | 1,669 | 9,447 | 5.6% |
| v13 | 4,906 | 1,319 | 1,996 | 1,669 | 9,890 | 13.3% |
| v14 | 4,906 | 3,407 | 1,996 | 1,669 | 11,978 | 28.4% |
| v15 | 4,906 | 3,463 | 1,996 | 1,669 | 12,034 | 28.8% |

These are row fractions under the current shuffled-row sampler, not guaranteed fractions of supervised tokens.

- UltraEdit generation (`build_ultraedit_teacher_pilot.py`) sends only source pixels and instruction to the teacher. SciForma generation (`build_editing_teacher_pilot.py`) does the same. Both are RFT-style supervision, despite saving the dataset target for evaluation. Neither has a complete TI corpus yet.
- Current MagicBrush TI starts from 5,826 accepted trajectories representing 5,044 canonical task IDs. It selects already-passed source-only tasks, rather than rescuing failed tasks as original STaR does.
- v12–v15 contain the same 9,147 canonical task IDs across train+val after stripping teacher suffixes. The row increase primarily adds trajectories; it should not be presented as adding that many new tasks.
- Existing TI selection keeps the first passing candidate by directory order; it does not pick the highest-quality teacher candidate.
- The flip probe records execution and change, but the SFT builder only checks execution status. It does not require nonzero change, correctness on the flipped image, or edit equivariance. It is a weak execution check, not proof of generalization.
- In v14, 174 canonical MagicBrush tasks occur across internal train/val; 432 MagicBrush image IDs occur across both. Teacher suffixes and editing turns were hashed separately. This invalidates internal validation as a held-out source estimate. The separate official MagicBrush dev image IDs have zero overlap with the training image IDs in this identifier audit; image-content/near-duplicate audit remains necessary for all datasets.
- The training manual prefix and the evaluation native chat template differ. A real v14 row produced 793 prompt tokens in training versus 796 in evaluation, first differing at token 52: image/text order differs and evaluation adds empty `<think>` tags. Prompt parity must be fixed and verified before final runs.
- The external SFT converter hardcodes `requires_inpaint=False`, even for synthesis instructions, and omits teacher identity. The SciForma pilot main call also fails to pass `args.teacher` into `process`, so current CLI names alone are insufficient evidence of backend identity. Repair provenance from raw traces/logs; do not infer it from filenames.

Current custom judge rates on 64 examples from the official dev split are observations: v12-fresh 14/64, v13 16/64, v14 20/64, v15 16/64. They are **not official benchmark success rates** or a controlled dose-response result. The v12-fresh input contains TI, total row counts and teacher mixtures differ, and continuation scripts reset optimizer, data order, and LR schedule. These observations neither establish saturation near 26% nor establish that SFT cannot improve further. Two extra successes out of 64 in the long run do not prove a significant duration benefit or additive effects.

## 4. Final data production recipe

### Shared task manifest and split first

Maintain `dataset, dataset_version, original_split, task_id, source_group_id, source_hash, instruction, target_path, mask_path`. Split before any generation. All candidates, teachers, modes, and turns for a source group inherit one split. For MagicBrush, group by original image ID; for SciForma, group by original figure/paper where available; otherwise use exact and perceptual source-image duplicate clusters. Compare all training datasets against all evaluation sources and targets.

Retain official test sets for final reporting. Use clean internal validation for recipe/checkpoint selection. Official development subsets already used during exploration are development data, not an untouched final test.

### Generate two candidate pools for all three datasets

1. RFT: teacher sees source + instruction; returns brief source-grounded evidence, operation plan, and `def edit(image)`.
2. TI: teacher sees source + instruction + target; returns a program derived from source-visible geometry and RGB. Target guides the desired edit, but the explanation may not use phrases such as “as seen in target image 2” as necessary inference evidence. No mask in the default prompt.
3. Per-task candidate IDs include dataset/mode/teacher/attempt. Use unique execution paths; current TI `/tmp` paths based only on task ID can collide across concurrent teachers.
4. The final student rows in both pools use the exact same source-only conversation template and output schema. Preserve visible teacher evidence/plan/code; do not claim access to private internal reasoning.

For a controlled pilot, assign the same teacher to the same task across modes and generate two independent candidates per mode. Different modes may have different image-token costs; report teacher calls and total generation token cost separately. Keep at most two distinct verified programs per task per mode; duplicates/no-op candidates do not inflate task counts. For broad production, additional teachers can be used, but teacher assignment and candidate-ranking rules must be frozen across ablations.

### Verify rather than merely parse

- JSON/schema, nonempty `edit` function, source-only execution, output image integrity and correct dimensions.
- Instruction adherence, preservation, visual quality: a frozen judge protocol; judge API failures are pending/infrastructure failures, not semantic negatives.
- Target-image agreement: benchmark-aligned image metrics and edit-region diagnostics, alongside no-op source baseline. A target metric alone is insufficient; a no-op can be very close to target when most pixels are unchanged. Do not require exact RGB identity for edits with several valid outcomes.
- Leakage: reject target/mask reads, external-file dependencies, encoded target-image dumps, and source-independent raster reconstruction. Ordinary source-derived coordinates and colors remain allowed. Check target-only details in explanations and plans.
- Reusability: for eligible transforms, compare execution on transformed source with transformed original output (equivariance); transform directional instructions consistently. Do not use horizontal flip as an unconditional semantic check for “left/right” edits.
- Reproducibility: fix interpreter/library versions and explicit RNG seeds, including `default_rng`/OpenCV where used; repeat execution and compare RGB/code hashes. Python/NumPy global seeds alone do not prove determinism of every generated program or of model decoding/judging.

Store immutable original prompt, teacher/model version, visible response, code, generation mode, target/mask exposure flags, judge version, metric results and sample lineage. Keep a stratified manual audit by dataset × construction mode × teacher; calibrate the judge on ground-truth targets, no-op outputs, successful edits and clearly failed edits.

### Proposed initial training mixture

Dataset sampling weights: MagicBrush 60%, UltraEdit 25%, SciForma 15%. Within **each** dataset, start at RFT 75% / TI 25% (proposal, not a proven optimum).

| Bucket | Sampling weight |
|---|---:|
| MagicBrush RFT | 45% |
| MagicBrush TI | 15% |
| UltraEdit RFT | 18.75% |
| UltraEdit TI | 6.25% |
| SciForma RFT | 11.25% |
| SciForma TI | 3.75% |

Retain all qualified candidates; sampling weights determine exposure. Do not duplicate easy successful tasks merely to fill a quota. If any bucket is short, report the shortage and collect more data; do not silently substitute another dataset or mode. Also measure realized supervised-token fractions because code lengths differ. The existing trainer only shuffles rows; it does not implement these bucket weights yet. Implement a seeded bucket sampler or materialize a reproducible sampling plan before running this recipe.

Model recipe: Qwen3.8-27B, standard language LoRA r16, assistant-only loss, one image + original instruction, one JSON answer with evidence/plan/code. Use one shared native serializer for training and inference, one thinking-mode setting, one image preprocessing setting, and matching code execution helpers. Start the controlled runs from the same instruct checkpoint and same clean environment. Restore optimizer/scheduler/RNG/data cursor for true resume; label adapter-only restarts as additional fine-tuning stages. Freeze the selected recipe and checkpoint before final test evaluation.

## 5. Ablations: separate mechanism, coverage and scale

### A. Main construction-mode comparison (four SFT runs)

Build a clean **matched-support** core: exact same tasks for which both RFT and TI have verified candidates. Freeze dataset weights (60/25/15), task coverage, teacher assignment, quality filter, prompt schema, initialization, sampling policy, optimizer/LR and effective batch. Change only the construction-mode mixture:

| Arm | RFT/TI | Question |
|---|---|---|
| A0 | 100/0 | Is independent teacher distillation enough? |
| A1 | 75/25 | Does a modest TI mixture help? Initial candidate recipe. |
| A2 | 50/50 | Does more TI help beyond A1? |
| A3 | 0/100 | Is pure TI sufficient, or does it harm generalization? |

Use length-stratified selection and equal supervised-token budgets for the primary comparison; disclose any residual instruction/length differences and actual optimizer steps. Add a fixed-step comparison only if length differences matter. For the winning mixed arm and RFT-only baseline, repeat training with three seeds before claiming a reliable gain. This core tests trace quality, but cannot test recovery of RFT-failed tasks by definition.

### B. STaR failure rescue (separate experiment)

Start from the full clean training task pool, not the accepted RFT pool. Compare, under a stated equal teacher-token budget:

- More RFT: source-only attempts with all the budget.
- STaR fallback: limited RFT first; spend the remaining budget on TI for tasks where source-only generation did not produce a qualified program; train on the union of source-only successes and TI rescues with hints removed.

Report recovered unique tasks, recovery quality, accepted programs per teacher token, and held-out performance. An additional task-count-matched run distinguishes more coverage from better supervision. Do not force an RFT-only row onto a task that has no passing RFT program. A later student STaR round must generate fresh candidates with the improved Qwen; training longer on old teacher rows is only continued SFT.

### C. Secondary ablations, ordered by value

1. **Code-only vs evidence+plan+code** on identical accepted programs/tasks; match code-token exposure and report added reasoning-token compute. This tests visible reasoning supervision.
2. **TI without mask vs TI with mask** on identical tasks/teachers, with the same leakage and output-quality filter. Tests whether the optional extra privileged input helps.
3. **Dataset contribution**: MB-only vs MB+UE vs MB+Sci vs all three, with equal training token budgets and fixed within-dataset RFT/TI ratio. Report all three held-out benchmarks, including cross-domain transfer. State that task domains intentionally change.
4. **Training scale**: winning recipe at 0.5×/1×/2× supervised-token exposure, with the same schedule policy and true resume; do not label interrupted adapter-only restarts as a clean duration ablation.
5. **Teacher diversity**: one fixed teacher vs verified multi-teacher candidates only after the main mode comparison; otherwise teacher identity confounds the RFT/TI result.

No need to run every Cartesian combination. Start with A0–A3 and B, select the recipe on clean validation, then code-only and dataset contribution. RL should start from a reproducible SFT checkpoint; current evidence supports proceeding with an RL pilot, not declaring SFT scientifically exhausted.

## 6. Evaluation and decision rule

Use the same held-out tasks for every compared model: MagicBrush official dev (full 528 for recipe evaluation if feasible), UltraEditBench, SciFormaBench; reserve official test or a genuinely untouched split for the final table. Historical 64-case subsets are smoke/development estimates. Align metric implementation/preprocessing with each benchmark rather than assuming every metric in `eval_official_metrics.py` is official.

Report separately: parse rate, execution rate, completed judge coverage, overall semantic pass, semantic pass conditional on execution, instruction/preservation/quality scores, benchmark target-image metrics, source no-op baseline, response tokens and time. Target CLIP/DINO gains are supplementary diagnostics, not synonymous with edit success. Compare target metrics on a common executed-case subset and disclose failures to prevent varying denominators hiding regressions.

Use paired per-task differences, paired bootstrap or McNemar for binary semantic success, and training-seed variability. On a 64-case sample around 25–30% success, uncertainty is large; 2–6 additional passes are a useful pilot signal, not proof of saturation, causality or a universal optimal TI ratio.

Final mixture selection: pick on clean validation; start with A1 (75/25), choose A2 or A3 only if replicated held-out results support it. Retain an equal-budget RFT baseline in the final paper table. An ICML claim should explain whether improvement comes from higher-quality programs, more recovered tasks, or different data domains.
