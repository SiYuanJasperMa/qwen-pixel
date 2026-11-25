"""Persistent one-shot RFT production, stronger-teacher retries and TI rescues."""
import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from pixel.persistent_io import atomic_text,retry_write
from pixel.data_language import english_task,english_supervision
from pixel.source_policy import disabled_sources,source_disabled

TI_TEACHERS = ('gpt55', 'gpt61', 'terra', 'luna', 'gpt62')
DATASETS = ('magicbrush', 'ultraedit', 'sciforma', 'anyedit', 'maskedit_scene',
            'maskedit_en', 'instructpix2pix')


def primary_ti_teacher(task_id, teachers=TI_TEACHERS):
    return teachers[int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(teachers)]


def off_peak(beijing):
    return beijing.weekday() >= 5 or beijing.hour >= 18 or beijing.hour < 9


def lane_caps(night, gpt62_only=False, ti_only=False, ti_teachers=TI_TEACHERS):
    if ti_only:
        return {('ti', teacher): (16 if teacher in ('gpt61', 'gpt62') and night else
                4 if teacher in ('gpt61', 'gpt62') else 8 if teacher == 'gpt55' else 4 if night else 2)
                for teacher in ti_teachers}
    if gpt62_only:
        return {('rft', 'gpt62'): 8 if night else 2, ('ti', 'gpt62'): 4 if night else 1}
    return {('rft', 'gpt55'): 8, ('rft', 'gpt61'): 16 if night else 4,
            ('rft', 'terra'): 4 if night else 2, ('rft', 'luna'): 4 if night else 2,
            ('ti', 'gpt61'): 8 if night else 4}


def rows(path):
    if not path.is_file(): return []
    result = []
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass
    return result


def atomic_json(path, value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2))


def primary_teacher(task_id):
    slot = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 10
    return 'gpt55' if slot < 4 else ('gpt61' if slot < 8 else ('terra' if slot == 8 else 'luna'))


def qualified(record):
    return (record.get('judge') or {}).get('pass') is True and not record.get('hardcoded') \
        and bool(record.get('code')) and record.get('execution', {}).get('status') == 'passed' \
        and record.get('reuse_exec') == 'passed' \
        and record.get('pixel_metrics', {}).get('changed_pixel_fraction', 1) > 0


def pending_error(record):
    return record.get('status') == 'teacher_error' or bool((record.get('judge') or {}).get('error'))


def next_job(task, latest, gpt62_only=False, ti_only=False, ti_teachers=TI_TEACHERS,
             candidate_tag='growth1'):
    if source_disabled(task):
        return None
    teacher = primary_teacher(task['id'])
    stages = [('rft', teacher)] + ([('rft', 'gpt61')] if teacher != 'gpt61' else []) + [('ti', 'gpt61')]
    if gpt62_only:
        stages = [('rft', 'gpt62'), ('ti', 'gpt62')]
    if ti_only:
        # Independent TI enrichment, including tasks already solved by RFT.
        primary = primary_ti_teacher(task['id'], ti_teachers)
        order = [primary] + [t for t in ('gpt62', 'gpt61', 'gpt55', 'terra', 'luna')
                              if t in ti_teachers and t != primary]
        stages = [('ti', teacher) for teacher in order]
    for mode, model in stages:
        key = f"{task['id']}__{mode}_{model}_{candidate_tag}"
        record = latest.get(key)
        if not record or pending_error(record): return key, mode, model
        if qualified(record): return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'data/editing_growth_v1'))
    parser.add_argument('--gpt62-only', action='store_true', help='Independent GPT-6.2 RFT/TI lane')
    parser.add_argument('--ti-only', action='store_true', help='Independent English TI enrichment; RFT success does not suppress TI')
    parser.add_argument('--ti-teachers', nargs='+', choices=TI_TEACHERS, default=list(TI_TEACHERS))
    parser.add_argument('--candidate-tag', default='growth1',
                        help='Immutable candidate namespace for independent additional responses')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS,
                        help='Restrict new production to selected source families')
    parser.add_argument('--off-peak-only', action='store_true',
                        help='Schedule new requests only outside Beijing working hours; never cancel in-flight calls')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.candidate_tag):
        parser.error('--candidate-tag must contain only letters, digits, underscores or hyphens')
    if args.gpt62_only and args.ti_only:
        parser.error('--gpt62-only and --ti-only are separate construction modes')
    out = Path(args.out)
    shared = ROOT / 'data/editing_growth_v1'
    if args.gpt62_only and out.resolve() == shared.resolve():
        out = shared / 'gpt62'
    if args.ti_only and out.resolve() == shared.resolve():
        out = shared / 'ti_balanced_en_v1'
    out.mkdir(parents=True, exist_ok=True)
    for name in ('raw', 'executions', 'reuse', 'records'):
        (out / name).mkdir(exist_ok=True)
    lock = (out / 'producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(out / 'pid.json', {'pid': os.getpid(), 'started': time.time()})
    visited = {r['id'] for name in ('bulk_retry_tasks.jsonl', 'bulk_retry_tasks_2.jsonl')
               for r in rows(ROOT / 'data/star_clean_v2' / name)}
    task_files = [ROOT / 'data/star_clean_v2/failed_train.jsonl',
                  (shared if args.gpt62_only or args.ti_only else out) / 'tasks_train.jsonl']
    tasks, latest, errors, inflight = {}, {}, collections.Counter(), {}
    for record in rows(out / 'judged.jsonl'):
        latest[record['id']] = record
        if pending_error(record): errors[record['id']] += 1
    for path in(out/'records').glob('*.json'):
        record=json.loads(path.read_text());latest[record['id']]=record
    loaded = {}
    policy_seen = None

    def refresh():
        nonlocal policy_seen
        disabled = disabled_sources()
        if disabled != policy_seen:
            loaded.clear();policy_seen = disabled
        for identity in list(tasks):
            if tasks[identity].get('dataset') in disabled:
                del tasks[identity]
        for path in task_files:
            if not path.is_file(): continue
            signature = (path.stat().st_size, path.stat().st_mtime)
            if loaded.get(str(path)) == signature: continue
            for task in rows(path):
                if task.get('dataset') in disabled:continue
                if task['id'] in visited or task.get('split') != 'train': continue
                if not english_task(task): continue
                if args.datasets and task.get('dataset') not in args.datasets: continue
                if not task.get('source_group_id'): raise ValueError('Missing audited source group')
                # A routing-only dry run must not stat thousands of NAS images.
                # Live production still checks both files before scheduling.
                if not args.dry_run and (not Path(task['image']).is_file() or
                                         not Path(task['target_image']).is_file()): continue
                tasks[task['id']] = task
            loaded[str(path)] = signature

    refresh()
    if args.dry_run:
        print(json.dumps({'tasks': len(tasks), 'datasets': dict(collections.Counter(t['dataset'] for t in tasks.values())),
                          'english_only': True, 'ti_only': args.ti_only,
                          'candidate_tag': args.candidate_tag, 'off_peak_only': args.off_peak_only,
                          'image_existence_check': 'deferred_to_live_refresh',
                          'primary_teachers': dict(collections.Counter(primary_ti_teacher(t,args.ti_teachers)
                              if args.ti_only else 'gpt62' if args.gpt62_only else primary_teacher(t) for t in tasks))})); return
    from build_magicbrush_target_informed import process, traex_key
    from build_teacher_magicbrush_inpaint import PROMPT, SYSTEM
    from pixel.teacher_api import judge_edit, make_client
    key = None if args.gpt62_only else traex_key(); client = make_client()
    log = (out / 'judged.jsonl').open('a', buffering=1)
    last_export = 0

    def execute(task, job):
        jid, mode, teacher = job
        if source_disabled(task):
            return {'id':jid,'task_id':task['id'],'dataset':task['dataset'],
                    'generation_mode':mode,'teacher':teacher,'instruction':task['instruction'],
                    'status':'source_disabled','source_group_id':task['source_group_id']}
        prior = latest.get(jid, {})
        if prior.get('retained_output') and (prior.get('judge') or {}).get('error'):
            record = dict(prior)
            try:
                record['judge'] = judge_edit(client, record['source_image'], record['retained_output'],
                                             record['instruction'], strict_extent=True)
                record['status'] = 'pass' if record['judge'].get('pass') else 'passed'
            except Exception as exc: record['judge'] = {'error': type(exc).__name__ + ': ' + str(exc)[:200]}
            return record
        sample = {**task, 'id': jid, 'task_id': task['id'], 'source_image': task['image'],
                  'rft_failure_observed': (task.get('rft_failure_observed', False) if args.ti_only
                                           else mode == 'ti' or task.get('rft_failure_observed', False))}
        return process(sample, out, key, client, teacher, False, mode,
                       strict_extent=True, keep_output=True,reuse_raw=True)

    def export():
        accepted = [r for r in latest.values() if qualified(r) and english_task(r) and english_supervision(r)]
        data = []
        for record in accepted:
            answer = {**record['reasoning'], 'executable_code': record['code'],
                      'requires_inpaint': record.get('requires_inpaint', False)}
            data.append({'id': record['id'], 'task': record['dataset'] + '_oneshot', 'system': SYSTEM,
                         'instruction': PROMPT.format(instruction=record['instruction']),
                         'image': record['source_image'], 'answer': json.dumps(answer, ensure_ascii=False),
                         'meta': {'task_id': record['task_id'], 'dataset': record['dataset'],
                                  'source_group_id': record['source_group_id'], 'split': 'train',
                                  'instruction': record['instruction'], 'teacher_backend': record['teacher'],
                                  'generation_mode': record['generation_mode'],
                                  'target_informed': record['generation_mode'] == 'ti',
                                  'judge_rubric': 'full_visible_target_v1',
                                  'pixel_metrics': record.get('pixel_metrics'),
                                  'construction_origin': 'independent_ti_en_v1' if args.ti_only else 'continuous_rft_ti_v1',
                                  'language_policy': 'english_only',
                                  'candidate_tag': args.candidate_tag,
                                  'ti_selection': 'independent_enrichment' if args.ti_only else 'failure_rescue'}})
        destination = out / 'sft/train.jsonl'; destination.parent.mkdir(exist_ok=True)
        atomic_text(destination,''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in data))
        atomic_json(out / 'sft/manifest.json', {'rows': len(data), 'source_groups': len({r['meta']['source_group_id'] for r in data}),
                    'datasets': dict(collections.Counter(r['meta']['dataset'] for r in data)),
                    'modes': dict(collections.Counter(r['meta']['generation_mode'] for r in data)),
                    'source': str(out / 'judged.jsonl'), 'frozen_existing_sft_unchanged': True,
                    'candidate_tag': args.candidate_tag, 'off_peak_only': args.off_peak_only,
                    'language_policy': 'english_only', 'ti_selection': 'independent_enrichment' if args.ti_only else 'failure_rescue'})

    export()
    print('PRODUCER_READY', len(tasks), flush=True)
    with cf.ThreadPoolExecutor(max_workers=64) as pool:
        while True:
            refresh()
            now = time.time()
            beijing = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
            night = off_peak(beijing)
            caps = lane_caps(night,args.gpt62_only,args.ti_only,args.ti_teachers)
            if args.off_peak_only and not night:
                caps = {lane: 0 for lane in caps}
            running = collections.Counter((job[1], job[2]) for _, job, _ in inflight.values())
            running_ids = {job[0] for _, job, _ in inflight.values()}
            ordered = sorted(tasks.values(), key=lambda t: hashlib.sha256(t['id'].encode()).hexdigest())
            if args.gpt62_only:
                # Prefer tasks without a qualified trace from the existing teachers.
                prior_sft = rows(shared / 'sft/train.jsonl')
                accepted_ids = {r['meta']['task_id'] for r in prior_sft}
                ordered.sort(key=lambda t: t['id'] in accepted_ids)
            pending = collections.Counter()
            for task in ordered:
                job = next_job(task, latest, args.gpt62_only, args.ti_only, args.ti_teachers,
                               args.candidate_tag)
                if not job: continue
                jid, mode, teacher = job; lane = (mode, teacher)
                pending[lane] += 1
                if jid in running_ids or running[lane] >= caps[lane]: continue
                prior = latest.get(jid, {})
                if pending_error(prior):
                    delay = min(3600, 30 * 2 ** min(errors[jid], 7))
                    if now - prior.get('recorded_at', 0) < delay: continue
                future = pool.submit(execute, task, job)
                inflight[future] = (task, job, now); running_ids.add(jid); running[lane] += 1
            if inflight:
                done, _ = cf.wait(inflight, timeout=15, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    task, job, started = inflight.pop(future)
                    try: record = future.result()
                    except Exception as exc:
                        record = {'id': job[0], 'task_id': task['id'], 'teacher': job[2], 'generation_mode': job[1],
                                  'dataset': task['dataset'], 'status': 'teacher_error',
                                  'instruction':task['instruction'],'source_group_id':task['source_group_id'],
                                  'error': type(exc).__name__ + ': ' + str(exc)[:300]}
                    record.update(recorded_at=time.time(), pipeline_qualified=qualified(record),
                                  candidate_tag=args.candidate_tag)
                    latest[record['id']] = record
                    if pending_error(record): errors[record['id']] += 1
                    atomic_json(out/'records'/f"{record['id']}.json",record)
                    retry_write(lambda:log.write(json.dumps(record,ensure_ascii=False)+'\n'))
                    print('RESULT', record['id'], record.get('status'), record['pipeline_qualified'], flush=True)
            else: time.sleep(15)
            if now - last_export > 120:
                export(); last_export = now
            lanes = []
            for lane, cap in caps.items():
                recs = [r for r in latest.values() if (r['generation_mode'], r['teacher']) == lane and
                        (english_task(r) or (not r.get('instruction') and english_task(tasks.get(r['task_id'],{}))))]
                lanes.append({'mode': lane[0], 'teacher': lane[1], 'concurrency': cap,
                              'running_or_waiting_api': sum((j[1], j[2]) == lane for _, j, _ in inflight.values()),
                              'pending': pending[lane], 'completed': sum(not pending_error(r) for r in recs),
                              'qualified': sum(qualified(r) and english_task(r) and english_supervision(r) for r in recs),
                              'retryable_errors': sum(pending_error(r) for r in recs)})
            accepted = [r for r in latest.values() if qualified(r) and english_task(r) and english_supervision(r)]
            atomic_json(out / 'status.json', {'updated': beijing.isoformat(), 'state': 'running' if inflight else
                        'waiting_for_off_peak' if args.off_peak_only and not night else 'waiting_for_new_tasks',
                        'pid': os.getpid(), 'night_parallelism': night, 'task_pool': len(tasks),
                        'disabled_sources':sorted(disabled_sources()),
                        'datasets': dict(collections.Counter(t['dataset'] for t in tasks.values())), 'lanes': lanes,
                        'accepted_trajectories': len(accepted), 'accepted_unique_tasks': len({r['task_id'] for r in accepted}),
                        'fully_processed_tasks': sum(next_job(t, latest, args.gpt62_only, args.ti_only,args.ti_teachers,
                                                            args.candidate_tag) is None for t in tasks.values()),
                        'language_policy': 'english_only', 'ti_selection': 'independent_enrichment' if args.ti_only else 'failure_rescue',
                        'teacher_models': list(args.ti_teachers) if args.ti_only else None,
                        'candidate_tag': args.candidate_tag, 'off_peak_only': args.off_peak_only,
                        'judge_rubric': 'full_visible_target_v1', 'sft_manifest': str(out / 'sft/manifest.json')})


if __name__ == '__main__': main()
