"""Aggregate shard eval results (shard*/results.jsonl) into a summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    for d in sys.argv[1:]:
        base = Path(d)
        rows = []
        for p in sorted(base.glob('shard*/results.jsonl')):
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
        if not rows:
            print(base, 'no rows')
            continue
        tasks = len(rows)
        executed = sum(1 for r in rows if (r.get('execution') or {}).get('status') == 'passed')
        passed = sum(1 for r in rows if (r.get('judge') or {}).get('pass'))
        summary = {'tasks': tasks, 'executed': executed, 'judge_pass': passed,
                   'overall_pass_rate': passed / tasks if tasks else 0,
                   'executed_pass_rate': passed / executed if executed else 0}
        (base / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(base.name, summary)


if __name__ == '__main__':
    main()
