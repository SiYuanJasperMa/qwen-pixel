import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('with_mask_review',[False,True])
def test_complete_trajectory_filter_with_and_without_mask_review(tmp_path,monkeypatch,with_mask_review):
    script=Path(__file__).resolve().parents[1]/'scripts/review_clean_references.py'
    spec=importlib.util.spec_from_file_location('reference_review_under_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    identity='clean_1_9_blur';task={'id':identity,'operation':'blur','original_scene_id':'1'}
    (tmp_path/'tasks.jsonl').write_text(json.dumps(task)+'\n')
    rows=[{'id':identity+f'_step_{i}','meta':{'trajectory_id':identity,'original_scene_id':'1'}}for i in range(4)]
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    reviews=tmp_path/'masks';reviews.mkdir()
    folder=tmp_path/'executions'/identity;folder.mkdir(parents=True)
    (folder/'source.png').write_bytes(b'fixed source receipt')
    (reviews/'1.json').write_text(json.dumps({'accepted':True,'source_task':identity,
        'source_sha256':hashlib.sha256((folder/'source.png').read_bytes()).hexdigest()}))
    calls=[]
    def fake_review(selected,data,out):
        calls.append(selected['id']);return {'id':selected['id'],'operation':'blur','judge':{'pass':True}}
    monkeypatch.setattr(module,'review',fake_review)
    argv=['review_clean_references.py','--data',str(tmp_path),'--all','--write-verified']
    if with_mask_review:argv+=['--mask-reviews',str(reviews)]
    monkeypatch.setattr(sys,'argv',argv);module.main()
    assert calls==[identity]
    assert len((tmp_path/'train_verified.jsonl').read_text().splitlines())==4
    summary=json.loads((tmp_path/'reference_review_manifest.json').read_text())
    assert summary['review_errors']==0
    assert summary['verified_native_images']==1


def test_acceptable_initial_edit_is_excluded_from_repair_supervision(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/review_clean_references.py'
    spec=importlib.util.spec_from_file_location('paired_reference_review_under_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=['needs_repair','already_acceptable']
    tasks=[{'id':identity,'operation':'blur','original_scene_id':str(i)}for i,identity in enumerate(ids)]
    (tmp_path/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in tasks))
    rows=[{'id':identity+f'_step_{i}','meta':{'trajectory_id':identity,'original_scene_id':identity}}
          for identity in ids for i in range(4)]
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    def fake_pair(task,data,out):
        return {'id':task['id'],'operation':'blur','judge':{'pass':True},
                'bad_judge':{'pass':task['id']=='already_acceptable'}}
    monkeypatch.setattr(module,'review_pair',fake_pair)
    monkeypatch.setattr(sys,'argv',['review_clean_references.py','--data',str(tmp_path),'--all','--write-verified','--require-bad-fail'])
    module.main()
    retained=[json.loads(l)for l in(tmp_path/'train_verified.jsonl').open()]
    assert len(retained)==4
    assert {r['meta']['trajectory_id']for r in retained}=={'needs_repair'}
    summary=json.loads((tmp_path/'reference_review_manifest.json').read_text())
    assert summary['constructed_bad_candidates_judged_acceptable']==1
