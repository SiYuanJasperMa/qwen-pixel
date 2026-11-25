#!/bin/bash
# Emergency quota relief: pause STaR generation and delete its transient images,
# plus unused external datasets and old eval artifacts.
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
pkill -9 -f "build_magicbrush_target_informed.py" 2>/dev/null
sleep 2
# TI judged.jsonl keeps code/reasoning; executions/reuse/raw are not needed for SFT.
rm -rf data/teacher_target_informed_*/executions data/teacher_target_informed_*/reuse data/teacher_target_informed_*/raw
# Unused external datasets (AnyEdit/MaskEdit/ChartX/SEED were low priority or dropped).
rm -rf data/external_edit/ChartX data/external_edit/AnyEdit data/external_edit/MaskEdit-10k data/external_edit/SEED-Data-Edit-Part1-Unsplash
# Partial official-dev eval outputs.
rm -rf outputs/official_dev_v3cont
# Old superseded outputs that survived the first cleanup.
rm -rf outputs/pixel_edit_demo outputs/imagenet_zero_shot_100 outputs/compare_coco_3tasks* outputs/compare_coco_grounding* outputs/compare_coco_qa_caption* outputs/gen_sft_v1_* outputs/eval_sft_v1_* outputs/zero_shot_qwen9b_* outputs/zero_shot_qwen38_27b_magicbrush_dev8* outputs/zero_shot_qwen38_27b_oneshot_dev32 outputs/sft_magicbrush_27b_v1 outputs/sft_magicbrush_27b_v2
echo done
