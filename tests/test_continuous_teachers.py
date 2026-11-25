import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continuous_editing_teachers import qualified, next_job, primary_teacher, primary_ti_teacher, off_peak, lane_caps
from pixel.data_language import english_task
import datetime as dt


def good_record():
    return {'judge': {'pass': True}, 'code': 'def edit(image): return image',
            'execution': {'status': 'passed'}, 'reuse_exec': 'passed',
            'pixel_metrics': {'changed_pixel_fraction': .2}}


def test_acceptance_requires_semantics_execution_and_actual_change():
    r = good_record(); assert qualified(r)
    r['judge']['pass'] = 'false'; assert not qualified(r)
    r = good_record(); r['pixel_metrics']['changed_pixel_fraction'] = 0; assert not qualified(r)
    r = good_record(); r['reuse_exec'] = 'failed'; assert not qualified(r)


def test_rft_failure_uses_stronger_teacher_then_target_informed():
    tid = next(f'task_{i}' for i in range(100) if primary_teacher(f'task_{i}') == 'gpt55')
    task = {'id': tid}; latest = {}
    jid, mode, teacher = next_job(task, latest)
    assert (mode,teacher) == ('rft','gpt55')
    latest[jid] = {'status': 'failed'}
    jid, mode, teacher = next_job(task, latest)
    assert (mode,teacher) == ('rft','gpt61')
    latest[jid] = {'status': 'failed'}
    assert next_job(task, latest)[1:] == ('ti','gpt61')
    latest[jid] = good_record()
    assert next_job(task, latest) is None


def test_queue_and_judge_errors_retry_same_stage_without_semantic_rescue():
    task = {'id': 'test_api_error'}
    jid, mode, teacher = next_job(task,{})
    assert next_job(task,{jid:{'status':'teacher_error'}}) == (jid,mode,teacher)
    assert next_job(task,{jid:{'status':'passed','judge':{'error':'rate limited'}}}) == (jid,mode,teacher)


def test_gpt62_lane_rescues_only_after_rft_failure_without_changing_legacy_routing():
    task = {'id': 'test_gpt62_lane'}
    legacy = next_job(task, {})
    rft = next_job(task, {}, gpt62_only=True)
    assert rft == ('test_gpt62_lane__rft_gpt62_growth1', 'rft', 'gpt62')
    assert next_job(task, {rft[0]: {'status': 'teacher_error'}}, gpt62_only=True) == rft
    assert next_job(task, {rft[0]: good_record()}, gpt62_only=True) is None
    assert next_job(task, {rft[0]: {'status': 'failed'}}, gpt62_only=True)[1:] == ('ti', 'gpt62')
    assert next_job(task, {rft[0]: good_record()}) == legacy


def test_independent_ti_is_not_suppressed_by_rft_success():
    task={'id':'paired_ti_task'}
    rft=next_job(task,{})
    ti=next_job(task,{rft[0]:good_record()},ti_only=True)
    assert ti[1:] == ('ti',primary_ti_teacher(task['id']))
    assert next_job(task,{ti[0]:good_record()},ti_only=True)is None
    alternate=next_job(task,{ti[0]:{'status':'failed'}},ti_only=True)
    assert alternate[1]=='ti'and alternate[2]!=ti[2]
    assert next_job(task,{ti[0]:{'status':'teacher_error'}},ti_only=True)==ti


def test_english_export_excludes_chinese_tasks_but_allows_scientific_unicode():
    assert english_task({'dataset':'sciforma','instruction':'Replace the Δ legend with a blue label.'})
    assert not english_task({'dataset':'maskedit_cn','instruction':'Change the heading color.'})
    assert not english_task({'dataset':'anyedit','instruction':'把汽车改成蓝色'})
    assert not english_task({'dataset':'anyedit','instruction':'Change the car color.','language':'zh'})
    assert english_task({'meta':{'dataset':'ultraedit','instruction':'Make the car blue.'}})


def test_off_peak_covers_after_work_and_weekends_and_increases_sol_ti():
    assert off_peak(dt.datetime(2026,11,25,18,0))
    assert not off_peak(dt.datetime(2026,11,25,12,0))
    assert off_peak(dt.datetime(2026,11,28,12,0))
    day=lane_caps(False,ti_only=True);night=lane_caps(True,ti_only=True)
    assert night[('ti','gpt61')]==16 and night[('ti','gpt62')]==16
    assert day[('ti','gpt61')]<night[('ti','gpt61')]
    assert night[('ti','gpt55')]==8


def test_additional_candidate_is_independent_and_does_not_overwrite_first_response():
    task = {'id': 'scientific_figure_train_example'}
    teachers = ('gpt61', 'gpt62')
    first = next_job(task, {}, ti_only=True, ti_teachers=teachers)
    latest = {first[0]: good_record()}
    assert next_job(task, latest, ti_only=True, ti_teachers=teachers) is None
    second = next_job(task, latest, ti_only=True, ti_teachers=teachers,
                      candidate_tag='night_sol_v2')
    assert second[0] != first[0]
    assert second[1:] == first[1:]
    latest[second[0]] = {'status': 'teacher_error'}
    assert next_job(task, latest, ti_only=True, ti_teachers=teachers,
                    candidate_tag='night_sol_v2') == second
    latest[second[0]] = good_record()
    assert next_job(task, latest, ti_only=True, ti_teachers=teachers,
                    candidate_tag='night_sol_v2') is None
