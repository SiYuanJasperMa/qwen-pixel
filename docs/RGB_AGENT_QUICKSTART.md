# Qwen Pixel: direct RGB agent quickstart

The current minimal interface has three actions: `python`, `view_image`, and `finish`.

In the `rgb` profile, Python receives `image` (the original PIL image) and `pixels` (a full-resolution, writable uint8 RGB array). The model computes edited pixels. `emit_rgb(array)` only rounds/clips to the 8-bit RGB domain and saves the model's array as `edited.png` and `edited_rgb.npy`; it does not locate objects, create masks, inpaint, or perform an edit.

For example, a model can compute a local change as:

```python
out = pixels.copy()
# The model chooses the region and transformation from the source and instruction.
out[y1:y2, x1:x2] = transformed_pixels
emit_rgb(out)
```

Actual source, crop, mask, and edited images returned by `view_image` are frozen under `observations/` with SHA256 filenames. A later edit cannot change an earlier image in the training prefix or RL replay.

## Task manifest

Provide real images and edit instructions in JSONL:

```json
{"id":"scene_001_edit_01","image":"images/scene_001.jpg","instruction":"Change the red bus bodywork to blue.","split":"train","original_scene_id":"scene_001"}
```

Keep every image/version of the same source scene in the same split. Official benchmark test records must stay outside teacher SFT and RL training.

## Teacher trajectories

```bash
python scripts/collect_agent_teachers.py --tasks tasks.jsonl \
  --out data/rgb_teachers --concurrency 8 --max-steps 8 \
  --effort low --system-profile rgb
python scripts/audit_agent_dataset.py data/rgb_teachers
```

The collector runs the teacher's real code, captures real feedback, checks a valid pixel submission, and judges instruction adherence and preservation. Only judged-pass trajectories enter `train.jsonl` and `val.jsonl`. Failed cases remain recorded separately. Configure the teacher API through the available client/environment; credentials are not dataset fields.

## SFT

```bash
python train/run_pixel_sft.py train --model "$QWEN_PIXEL_MODEL" \
  --data data/rgb_teachers/train.jsonl --val data/rgb_teachers/val.jsonl \
  --output outputs/rgb_sft --steps 300 --accum 2 --rank 16 --lr 7e-5 \
  --gradient-checkpointing --max-seq-len 8192 --image-max-pixels 262144 \
  --save-every 300 --nonzero-weight 1
```

Each record supervises only the next assistant operation. Previous assistant turns, tool text, and returned images are context. Target/future images do not appear before their actual tool return.

## RL and evaluation

```bash
python rl/run_agent_grpo.py --model "$QWEN_PIXEL_MODEL" \
  --adapter outputs/rgb_sft/checkpoint-300 --tasks tasks.jsonl \
  --out outputs/rgb_grpo --groups 2 --group-size 3 \
  --max-steps 8 --system-profile rgb
python scripts/eval_agent_tasks.py --model "$QWEN_PIXEL_MODEL" \
  --adapter outputs/rgb_grpo/final_adapter --tasks tasks.jsonl \
  --out outputs/eval_rgb --max-steps 8 --system-profile rgb
```

GRPO samples full tool trajectories and scores the actual sampled assistant token IDs. It uses a group-normalized outcome reward and a frozen-base KL reference. Incorrect targets or destroyed unrelated content get zero reward even if a file was successfully written. Equal-reward groups are skipped; reporting a successful RL smoke requires nonzero gradients and changed adapter parameters.

Compare Instruct, SFT, and RL with identical images, instructions, tool interfaces, image token budgets, and step budgets. Report valid submission and semantic editing success separately. A one-task overfit demonstration proves the execution/training interface; it does not measure generalization.
