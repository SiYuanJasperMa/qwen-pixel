import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pixel import source_policy
from pixel.data_quality import snapshot_quality_reason
from continuous_editing_teachers import next_job


@pytest.mark.parametrize('row',[
    {'dataset':'instructpix2pix','id':'x'},
    {'meta':{'dataset':'InstructPix2Pix','task_id':'x'}},
    {'id':'instructpix2pix_s0_1092__rft_gpt55_growth1'},
    {'dataset':'mixed','origin':{'repo':'timbrooks/instructpix2pix-clip-filtered'}}])
def test_disabled_source_is_recognized_across_task_export_and_archive_forms(row):
    assert source_policy.source_disabled(row)


def test_disabled_task_never_gets_rft_retry_rescue_or_independent_ti():
    task={'id':'instructpix2pix_s0_1092','dataset':'instructpix2pix'}
    for settings in [{},{'gpt62_only':True},{'ti_only':True}]:
        assert next_job(task,{},**settings)is None
    assert snapshot_quality_reason({'meta':task},{'excluded_tasks':{}})=='source_disabled_by_user'
    assert next_job({'id':'magicbrush_train_example','dataset':'magicbrush'}, {})is not None


def test_teacher_entry_point_does_not_open_images_or_call_api_for_disabled_source(tmp_path,monkeypatch):
    import build_magicbrush_target_informed as teacher
    def no_api(*args,**kwargs):
        raise AssertionError('Disabled source must not reach teacher API')
    monkeypatch.setattr(teacher,'call_multi',no_api)
    result=teacher.process({'id':'instructpix2pix_s0_1092','dataset':'instructpix2pix'},
                           tmp_path,None,None,'gpt55',False,'rft')
    assert result['status']=='source_disabled'
    assert list(tmp_path.iterdir())==[]


def test_opt_out_remains_when_optional_policy_file_is_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(source_policy,'ROOT',tmp_path)
    assert 'instructpix2pix'in source_policy.disabled_sources()


def test_ingester_default_source_list_excludes_disabled_dataset():
    from ingest_editing_datasets import SOURCES
    assert all(source[0]!='instructpix2pix'for source in SOURCES)


def test_new_paired_snapshots_cannot_reintroduce_archived_disabled_source():
    from build_paired_method_sft import answer_allowed
    assert answer_allowed({'id':'archived','meta':{'dataset':'instructpix2pix',
        'task_id':'instructpix2pix_s0_1092'}})==(False,'source_disabled_by_user')
