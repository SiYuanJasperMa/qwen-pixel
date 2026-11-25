> **2026-11-22 核查更正：最终方案见 `SFT_FINAL_DESIGN.md`。**
> 当前是 STaR-inspired teacher TI distillation，不是完整的 STaR 迭代：TI 输入池为已通过 RFT 的任务。翻转探针只要求代码可执行，并未筛除所有 no-op 或证明语义可复用；下面历史描述中更强的保证不成立。内部划分和训练/推理模板均有待修复。

# STaR / Target-Informed Distillation Pipeline

## Why

Source-only teacher trajectories are semantically plausible but often **not closer to
the dataset target than doing nothing**: only 30% have positive L1 gain, and even
GPT-6.1/GPT-5.5 have negative CLIP gain on MagicBrush dev32. STaR fixes this by giving
the teacher the ground-truth target at **data-generation time only**.

## Pipeline

```
MagicBrush accepted sample (source, target, mask, instruction)
  -> teacher sees source + target (+ mask), writes def edit(image)
  -> parse JSON, execute on source, VLM judge (pass/fail)
  -> hardcoded-code filter + flip-reuse probe
  -> one-shot SFT row: (source + original instruction) -> (reasoning + code)
```

### 1. Input

`data/teacher_magicbrush_merged_v1/judged.jsonl` (5,826 accepted MagicBrush samples),
each with `source_image`, `target_image`, `mask_image`, `instruction`.

### 2. Teacher call

`scripts/build_magicbrush_target_informed.py`

- Images: Image 1 = source, Image 2 = target, Image 3 = mask (optional `--use-mask`).
- Prompt: analyze the difference, write complete runnable code that transforms source
  toward target.
- Hard requirements: code runs with **only the source**; must not read target/mask/external
  files; no hardcoded pixels/coordinates; preserve unrelated regions; define
  `def edit(image)` returning a PIL Image.
- Teachers: `gpt55`, `deepseek`, `gpt61` (+mask), `luna` (GPT-5.6-Sol route was 502).

### 3. Verify

- Parse JSON, extract `executable_code`.
- Execute on the source (`run_code`, 60s timeout).
- Judge with GPT-5.5: source + result + instruction -> instruction adherence,
  preservation, visual quality, pass/fail.
- Output `data/teacher_target_informed_<teacher>/judged.jsonl`.

### 4. Leakage guards

| Guard | Rule |
|---|---|
| hardcoded filter | reject file references, target/mask path variables, huge literal arrays, long numeric runs, constant pixel writes |
| flip-reuse probe | run the code on a horizontally flipped source; must still execute and change the image |

### 5. SFT conversion

`scripts/build_target_informed_sft.py`

- Keep only judge-pass + non-hardcoded + flip-reuse-passed, dedup by sample id.
- Input: system + source image + **original instruction** (same as inference).
- Output: teacher reasoning JSON + `executable_code`.
- Hash split 10% val.
- Output `data/sft_target_informed_v*/{train,val}.jsonl`.

## Current results (2026-11-20)

| teacher | generated | judge pass | hardcoded | reuse pass |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 1,435 | ~34% | 0.1% | 84% |
| GPT-5.5 | 2,618 | ~32% | 3.2% | 79% |
| GPT-6.1-Sol + mask | 798 | ~38% | 36.7% | 80% |
| Luna | 155 | running | — | — |

Target fidelity (50 accepted samples per teacher, re-executed):

| teacher | TI L1 | base L1 | TI gain | base gain | TI gain>0 |
|---|---|---|---|---|---|
| DeepSeek | 0.0212 | 0.0263 | +0.00001 | -0.0051 | 34% |
| GPT-5.5 | 0.0185 | 0.0263 | +0.00069 | -0.0071 | 36% |
| GPT-6.1+mask | 0.0186 | 0.0231 | +0.00312 | -0.0014 | 76% |

TI data is consistently closer to the target than the source-only teacher data; the
gain flips from negative to positive.

## Training usage

- `v12` (9,447 rows) contains 532 TI rows (5.6%) — not a clean STaR test.
- `sft_ti_only_v1`: 903 TI-only rows, 600 steps, evaluated at ckpt-300/600.
- `v13` (next): `v12 + full TI` (~12-13k) once TI reaches 2,500-3,500 accepted.

## Caveats

- Target leakage: mitigated by hardcoded filtering and the flip probe, but not fully
  eliminated; held-out-source evaluation is the final check.
- Code is a lossy medium: TI cannot reproduce the target exactly (e.g. adding a specific
  dog), so gains are smaller than the judge-pass improvement.
- Judge noise: the VLM judge is the gate; sample audits are recommended.
- RL with target-based reward remains the way to go beyond the teacher ceiling.
