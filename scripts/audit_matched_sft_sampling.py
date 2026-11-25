"""Verify actual sample order, supervised-token counts and LR in matched SFT."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    for name in ['left-data','right-data','left-log','right-log','out']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    left, right = rows(args.left_data), rows(args.right_data)
    if len(left) != len(right):
        raise ValueError('Datasets have different row counts.')
    paired = {}
    for a, b in zip(left, right):
        if a['target_message'] != b['target_message'] or a.get('target_token_ids') != b.get('target_token_ids'):
            raise ValueError('Supervised targets are not matched.')
        if b['id'] in paired:
            raise ValueError('Duplicate sample identity.')
        paired[b['id']] = a['id']
    a = {r['step']:r for r in rows(args.left_log)}
    b = {r['step']:r for r in rows(args.right_log)}
    common = sorted(a.keys() & b.keys())
    mismatches = []
    for step in common:
        ar, br = a[step], b[step]
        matches = {'sample_order':ar['sample_ids']==[paired[s] for s in br['sample_ids']],
                   'supervised_tokens':ar['supervised_tokens']==br['supervised_tokens'],
                   'learning_rate':ar['lr']==br['lr'],
                   'target_actions':ar['sample_actions']==br['sample_actions']}
        if not all(matches.values()):
            mismatches.append({'step':step, 'matches':matches})
    report = {'paired_rows':len(left), 'common_steps':len(common),
              'last_common_step':common[-1] if common else None,
              'actual_sample_order_token_counts_and_lr_match':bool(common) and not mismatches,
              'mismatches':mismatches,
              'scope':'observed training prefix only; losses/gradients intentionally differ with histories; not image quality'}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if mismatches:
        raise ValueError('Actual training exposure is not matched.')


if __name__ == '__main__':
    main()
