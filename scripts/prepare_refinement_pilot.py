"""Prepare a bounded repair-aware SFT dataset from already verified photos."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'data/sft_clean_mask_v4_raw';repair=ROOT/'data/sft_semantic_repair_balanced_v2'
    out=ROOT/'data/sft_refinement_pilot_v1';out.mkdir(exist_ok=True)
    if (out/'train.jsonl').exists():raise ValueError('Refinement pilot already prepared; do not alter active data.')
    review=json.loads((repair/'reference_review_manifest.json').read_text())
    if not review['require_actual_bad_image_to_fail']or review['review_errors']:
        raise ValueError('Need strict paired bad/good image verification.')
    normal=[json.loads(l)for l in(base/'train_verified.jsonl').open()]
    repairs=[json.loads(l)for l in(repair/'train_verified.jsonl').open()]
    allowed={r['meta']['original_scene_id']for r in normal}
    if not {r['meta']['original_scene_id']for r in repairs}<=allowed:raise ValueError('Repair images are outside the established training cohort.')
    # 24% of sampled actions teach correction; all wrong actions remain context only.
    weighted=[]
    for repeat in range(2):
        for row in repairs:
            copied=json.loads(json.dumps(row));copied['id']+=f'_sampling_repeat_{repeat}'
            weighted.append(copied)
    combined=normal+weighted
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in combined))
    shutil.copyfile(base/'development_tasks.jsonl',out/'development_tasks.jsonl')
    manifest={'native_training_images':len(allowed),'normal_assistant_actions':len(normal),
        'unique_repair_actions':len(repairs),'repair_sampling_multiplier':2,'training_actions':len(combined),
        'new_training_photos':0,'paired_bad_fail_good_pass_required':True,
        'base_data_sha256':hashlib.sha256((base/'train_verified.jsonl').read_bytes()).hexdigest(),
        'repair_data_sha256':hashlib.sha256((repair/'train_verified.jsonl').read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest(),
        'plan':'Use after the matched standard/attention comparison; prepared only, not yet trained.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('REFINEMENT_PILOT_PREPARED',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
