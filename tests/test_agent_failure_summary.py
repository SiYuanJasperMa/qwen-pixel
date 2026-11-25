import importlib.util
import json
from pathlib import Path


def test_pending_judge_and_overlapping_failure_causes_are_distinct(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/summarize_agent_edit_failures.py'
    spec=importlib.util.spec_from_file_location('agent_failures',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'id':'pending','complete':True},
          {'id':'bad','complete':True,'judge':{'pass':False,'instruction_adherence':4,'preservation':2,'visual_quality':2,'target_extent_score':5,'target_extent_complete':True}}]
    for row in rows:
        folder=tmp_path/row['id'];folder.mkdir();(folder/'evaluation.json').write_text(json.dumps(row))
    r=module.summarize(tmp_path)
    assert r['counts']['complete']==2 and r['counts']['judged']==1 and r['counts']['judge_pending']==1
    assert r['counts']['preservation_fail']==1 and r['counts']['visual_quality_fail']==1
    assert 'semantic_pass'not in r['counts']


def test_repeated_calls_reported_separately_from_success(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/summarize_agent_edit_failures.py'
    spec=importlib.util.spec_from_file_location('agent_failures',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=tmp_path/'repeated';folder.mkdir()
    (folder/'evaluation.json').write_text(json.dumps({'id':'repeated','complete':False}))
    (folder/'trajectory.json').write_text(json.dumps({'turns':[
        {'action':'view_image','arguments':{'path':'source.png'}},
        {'action':'python','arguments':{'code':'emit_rgb(pixels)'}},
        {'action':'python','arguments':{'code':'emit_rgb(pixels)'}},
        {'action':'view_image','arguments':{'path':'edited.png'}}]}))
    report=module.summarize(tmp_path)
    assert report['counts']['repeated_identical_call_cases']==1
    assert report['cases'][0]['repeated_identical_calls']==1
    assert report['cases'][0]['causes']==['execution_incomplete']
