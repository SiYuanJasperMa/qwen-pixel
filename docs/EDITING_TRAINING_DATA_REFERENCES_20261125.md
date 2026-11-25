# Image-edit training data: current sources and paper references

Read-only research requested by the user. The SFT goal remains paused; no new
training or data-generation job was launched for this note.

## Current project data

The current English supervision inventory, combining the historical train export
and the new growth exports and excluding disabled InstructPix2Pix, has18,500
rows at this audit. These are candidate/archived supervision rows before final
quality selection and identical-supervision deduplication, not unique tasks or
all data used by one trained checkpoint.

| Source | English rows | Scope |
|---|---:|---|
| MagicBrush |8,826|Human-authored image editing instructions and reviewed edits; photo editing core|
| UltraEdit |4,304|Our imports use UltraEdit_Region_Based_100k; region-aware photo editing|
| AnyEdit |1,397|Source+instruction tasks; visual-reference-only shards are excluded|
| MaskEdit scene |983|Scene/photo edits|
| MaskEdit English infographic |986|English text/layout/infographic edits; CN excluded|
| SciForma Editing |2,004|Scientific diagram editing pathway; generation pathway not included|

The trained historical12,006-row main snapshot is MagicBrush8,286(69.02%),
UltraEdit2,038(16.97%) andSciForma1,682(14.01%). Those are inventory-derived row
fractions, not an optimal published mixture.

The later same-task/same-teacher256-taskRFT/TI pilot selected23MagicBrush,
46UltraEdit,26AnyEdit,46MaskEdit scene,46MaskEdit English,23SciForma and46
InstructPix2Pix tasks. Its frozen historical data stays intact. Future snapshots
excludeInstructPix2Pix, per `data/editing_source_policy.json`. Source fractions,
task fractions and supervised-token exposure are distinct quantities.

## Direct multi-source mixture reference: EditVerse

[EditVerse,2025](https://arxiv.org/html/2509.20360v1), AppendixTable10, lists
the following in its **image editing** section. Other generation/video training
branches are outside the present project scope.

| Source | Used examples | Published Ratio column |
|---|---:|---:|
| MagicBrush |8,802|10|
| ShareGPT-4o-Image |46,489|10|
| OmniEdit |185,500|2|
| ImgEdit |245,986|1|
| NHR-Edit |358,463|5|
| UltraEdit |500,000|1|
| AnyEdit |1,244,033|1|
| GPT-Image-Edit-1.5M |1,500,000|1|

The table also includes118,972 internal object-removal/addition examples with
Ratio4 and1,824,969 internal instruction-editing examples withRatio1. Do not
readRatio10 as10%, or reconstruct actual token fractions without the sampler.
The useful pattern is higher relative weight for small high-quality collections
and selective use of large generated sources. Table1 reportsAnyEdit2.5M before
filtering and about1.2M used; OmniEdit1.2M before filtering and186K used. The
authors explicitly describe noise in some categories even in newer datasets.

## Task-composition and quality reference: OmniEdit

[OmniEdit,ICLR2025](https://arxiv.org/html/2411.07199v2), Table4, gives5.2M
pre-filtering entries and1.22M post-filtering entries:

| Task | Post-filter entries | Corpus fraction |
|---|---:|---:|
| Object property modification |250,000|20.49%|
| Object addition |200,000|16.39%|
| Object removal |200,000|16.39%|
| Object replacement |180,000|14.75%|
| Background swap |180,000|14.75%|
| Environment change |160,000|13.11%|
| Style transfer |50,000|4.10%|

Seven task-specific specialists synthesize candidates. GPT-4o scores semantic
consistency and perceptual quality; its scoring supervision is distilled into
InternVL2-8B. A score>=9/10 is retained. Table4 is the source for these counts;
the prose's approximate“top10%” wording does not equal every task's measured
retention fraction. Their image editor is trained on image targets, not our
Python-code targets, so borrow coverage/filtering design rather than its loss.

## Other relevant construction recipes

**UltraEdit,2024:**
[Paper](https://arxiv.org/html/2407.05282), Section2.2 andAppendixB.
4,108,262 examples include4,000,083 free-form/no-region-annotation entries and
108,179 region-annotated entries;757,879 unique edits are reported. “No region
annotation” does not mean every edit is global. Real-image anchors, image-grounded
instructions, region extraction and quality filtering address weaknesses of
caption-only synthesis. Our imported100K region subset is not the full4M corpus.

**GPT-Image-Edit-1.5M,2025:**
[Paper](https://arxiv.org/html/2507.21033), Section3 andAppendixA.
MergesOmniEdit/HQ-Edit/UltraEdit, regenerates outputs withgpt-image-1 and selectively
rewrites instructions. Its release contains original/rewritten/complex instruction
variants andHQ-Edit dual branches; do not conflate all release records with unique
editing tasks. The current[repository](https://github.com/wyhlovecpp/GPT-Image-Edit)
announces a2026“GPT-Image-Edit-1M”revision; the construction statements here cite
the2025v1 paper and do not transfer old counts to the newer revision.

**AnyEdit,CVPR2025:**
[Paper](https://arxiv.org/abs/2411.15738) and
[project](https://dcd-anyedit.github.io/).
2.5M pairs,25 editing types across local/global/camera/implicit/visual categories.
Task-dependent generation and selection improve coverage. Visual categories
require additional reference/control inputs in some cases; they cannot all be
imported into a source-image+instruction-only interface.

**SciForma,2026:**
[Paper](https://arxiv.org/html/2607.18091v1), Sections4.2/4.4 andAppendixA.
Despite the700K name, Table7 lists656,000 generation pairs+70,000 editing
triplets=726,000 total training entries. The editing pathway has70K retained
triplets from252K processed candidates. Component/Arrow/Text axes provide an
explicit structure inventory, and71.4% of edits modify2–3 elements. Our project
uses theeditingpath, with separate text/topology preservation evaluation.

**GIER,2020,not currently imported:**
[A Benchmark and Baseline for Language-Driven Image Editing](https://arxiv.org/abs/2010.02330), Section3.2.
6,179 image pairs, five language requests per pair, editing-operation andmask
annotations;3.21 operations/pair on average. This is relevant for practical
global/local retouching and interpretable RGB programs. Operation categories and
masks do not constitute ready-made executable code or verified parameters.

## Implications for Qwen Pixel-Edit

Use independent axes forsource family,editingtype andRFT/TI construction mode.
Photo editing sources provide object/attribute/background/environment coverage;
MaskEdit English andSciForma provide practical text/diagram tasks. Select clear
instructions and source-target consistency, then retain only actually executable,
instruction-correct,preserved,high-quality teacher RGB outcomes. Preserve raw
rejections for auditing rather than declaring an entire newer source high-quality.

Separate data-size fractions from sampling weights and report actual assistant
tokens. Comparephoto-only versusphoto+diagram/text and source contributions under
equal training budgets, before fixing a final mixture. Editing papers generally
do not establish a universal RFT/TI ratio for language-model code distillation;
that requires our own matched experiments.
