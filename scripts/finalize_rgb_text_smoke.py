"""Finish evaluation of saved one-shot outputs without new model generations."""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.rgb_text_eval import pixel_metrics, summarize
from pixel.teacher_api import make_client, judge_edit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--tasks', default=str(ROOT / 'data/rgb_text_smoke_v1/tasks.jsonl'))
    parser.add_argument('--snapshot', action='store_true', help='Write sidecar evaluation while inference continues')
    args = parser.parse_args()
    folder = Path(args.out)
    tasks = {r['id']: r for r in map(json.loads, Path(args.tasks).read_text().splitlines())}
    records = list(map(json.loads, (folder / 'results.jsonl').read_text().splitlines()))
    cached_path = folder / 'evaluated_results.jsonl'
    cached = { (r['id'],r['mode']): r for r in map(json.loads,cached_path.read_text().splitlines()) } if cached_path.is_file() else {}
    records = [cached.get((r['id'],r['mode']),r) if cached.get((r['id'],r['mode']),{}).get('source_rgb_sha256') == r.get('source_rgb_sha256') else r for r in records]
    client = make_client()

    def finish(record):
        execution = record.get('execution', {})
        if execution.get('status') != 'passed':
            return record
        task = tasks[record['id']]
        record['metrics'] = pixel_metrics(task['image'], task['target_image'], execution['output'])
        if record.get('error', '').startswith("AttributeError: 'NoneType'"):
            record['evaluation_fix'] = 'Legacy mask-dependent metrics replaced by aligned RGB metrics.'
            del record['error']
        if record.get('judge', {}).get('rubric_version') == 'full_visible_target_v1':
            return record
        try:
            record['judge'] = judge_edit(client, task['image'], execution['output'],
                                        task['instruction'], strict_extent=True)
            record.pop('judge_pending', None)
            record.pop('judge_error', None)
        except Exception as exc:
            record['judge_pending'] = True
            record['judge_error'] = type(exc).__name__ + ': ' + str(exc)[:200]
        print('FINALIZED', record['id'], record['mode'], record.get('judge', {}).get('pass'), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(finish, records))
    # Sidecars support judging saved outputs while inference is still appending.
    result_name = 'evaluated_results.jsonl' if args.snapshot else 'results.jsonl'
    summary_name = 'evaluated_summary.json' if args.snapshot else 'summary.json'
    (folder / result_name).write_text(''.join(json.dumps(r) + '\n' for r in records))
    audit = json.loads((folder / 'model_audit.json').read_text())
    (folder / summary_name).write_text(json.dumps(summarize(records, audit), indent=2))


if __name__ == '__main__':
    main()
