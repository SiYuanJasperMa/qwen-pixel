"""Wait for the owned data builder, verify all reference edits, then run SFT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(arguments):
    subprocess.run([sys.executable,*arguments],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--builder-pid',type=int,required=True)
    parser.add_argument('--variant',default='sft_clean_mask_v3')
    parser.add_argument('--expected-images',type=int,default=40)
    parser.add_argument('--train-gpu',type=int,default=0)
    parser.add_argument('--audit-masks',action='store_true');args=parser.parse_args()
    try:
        fd=os.pidfd_open(args.builder_pid)
    except ProcessLookupError:
        fd=None
    if fd is not None:
        print('WAIT_OWNED_BUILDER',args.builder_pid,flush=True)
        select.select([fd],[],[]);os.close(fd)
    data=ROOT/'data'/args.variant
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['native_training_images']!=args.expected_images or not manifest['actual_reference_programs_executed']:
        raise ValueError('Expected a completely rebuilt reference dataset at the declared image count.')
    frozen=ROOT/'data/sft_clean_mask_v2/development_tasks.jsonl'
    shutil.copyfile(frozen,data/'development_tasks.jsonl')
    train_ids={str(q['image_id'])for q in manifest['quality']}
    development_ids={json.loads(l)['original_scene_id']for l in frozen.open()}
    final_ids={str(json.loads(l)['image_id'])for l in(ROOT/'data/independent_coco_edit_eval_v2/tasks.jsonl').open()}
    if train_ids&(development_ids|final_ids):raise ValueError('Evaluation image entered training.')
    (data/'split_audit.json').write_text(json.dumps({'train_images':len(train_ids),
        'development_images':len(development_ids),'final_images':len(final_ids),
        'train_evaluation_overlap':0,'development_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},indent=2))
    review_args=['scripts/review_clean_references.py','--data',str(data),'--all','--concurrency','8','--write-verified']
    if args.audit_masks:
        run(['scripts/audit_clean_instance_masks.py','--data',str(data),'--concurrency','8'])
        audit=json.loads((data/'mask_reviews/summary.json').read_text())
        if audit['review_errors']:raise ValueError('Resume failed semantic mask reviews before reference filtering.')
        review_args+=['--mask-reviews',str(data/'mask_reviews')]
    run(review_args)
    quality=json.loads((data/'reference_review_manifest.json').read_text())
    if quality['verified_native_images']<20:raise ValueError('Too few correctly edited reference images for this SFT pilot.')
    run(['scripts/audit_clean_training_rows.py','--data',str(data/'train_verified.jsonl')])
    run(['scripts/audit_sft_dataset_budget.py','--data',str(data/'train_verified.jsonl')])
    run(['scripts/render_sft_audit_page.py'])
    print('VERIFIED_SFT_TRAINING_START',args.variant,json.dumps(quality),flush=True)
    subprocess.run(['bash','scripts/train_clean_sft.sh',args.variant],cwd=ROOT,
        env=dict(os.environ,QWEN_PIXEL_TRAIN_GPU=str(args.train_gpu)),check=True)
    run(['scripts/render_sft_audit_page.py'])
    print('VERIFIED_SFT_TRAINING_FINISHED; inspect actual development edits before independent20.',flush=True)


if __name__=='__main__':main()
