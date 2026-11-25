"""Compare verified one-shot RFT retries against TI on identical task IDs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_magicbrush_target_informed import looks_hardcoded, PROGRAM_FILTER_VERSION


def records(path):
    return [json.loads(l) for l in (Path(path) / 'judged.jsonl').read_text().splitlines() if l.strip()]


def accepted(rows):
    return {r['task_id']: r for r in rows if (r.get('judge') or {}).get('pass')
            and r.get('code') and not looks_hardcoded(r['code'])[0]
            and r.get('reuse_exec') == 'passed'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rft', required=True); p.add_argument('--ti', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rr, tt = records(args.rft), records(args.ti)
    r, t = accepted(rr), accepted(tt)
    ids_r, ids_t = {x['task_id'] for x in rr}, {x['task_id'] for x in tt}
    common = ids_r & ids_t
    r = {k: v for k, v in r.items() if k in common}
    t = {k: v for k, v in t.items() if k in common}
    rescue = set(t) - set(r)
    union = list(r.values()) + [t[k] for k in sorted(rescue)]
    report = {'compared_tasks': len(common), 'rft_accepted': len(r), 'ti_accepted': len(t),
              'program_filter_version': PROGRAM_FILTER_VERSION,
              'both_accepted': len(set(r) & set(t)), 'ti_only_rescues': len(rescue),
              'rft_only_successes': len(set(r) - set(t)),
              'strict_star_union': len(union), 'rescued_task_ids': sorted(rescue),
              'rft_incomplete': len(ids_t - ids_r), 'ti_incomplete': len(ids_r - ids_t)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    (out / 'strict_star_judged.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in union))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
