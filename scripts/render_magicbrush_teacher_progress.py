"""Render the live MagicBrush SFT construction dashboard focused on the 10k target."""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')
SCALE = ROOT / 'data/teacher_magicbrush_scale_v1'
TARGET_TRAJECTORIES = 5000
TARGET_ROWS = 10000
SHARDS_GLOB = '/mnt/bn/embodied-lf3/masiyuan/qwen_image_edit_smoke/data/magicbrush_hf/data/train-*.parquet'
LEGACY = [
    ('GPT-6.1-Sol', ROOT / 'data/teacher_magicbrush_gpt61_inpaint_54_v1'),
    ('GPT-5.5', ROOT / 'data/teacher_magicbrush_gpt55_inpaint_54_v1'),
    ('GPT-5.6-Terra', ROOT / 'data/teacher_magicbrush_terra_inpaint_54_v1'),
    ('GPT-5.6-Luna', ROOT / 'data/teacher_magicbrush_luna_inpaint_54_v1'),
]


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_size():
    cache = ROOT / 'outputs/magicbrush_teacher_progress/dataset_size.json'
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get('shards') == len(glob.glob(SHARDS_GLOB)):
                return data
        except Exception:
            pass
    shards = sorted(glob.glob(SHARDS_GLOB))
    rows = 0
    for shard in shards:
        rows += pq.ParquetFile(shard).metadata.num_rows
    data = {'shards': len(shards), 'rows': rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def parse_time(value):
    try:
        return dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        return None


def scale_state():
    accepted = load_jsonl(SCALE / 'judged.jsonl')
    attempts = load_jsonl(SCALE / 'attempts.jsonl')
    per = {}
    for row in attempts:
        teacher = row.get('teacher', 'unknown')
        entry = per.setdefault(teacher, {'attempts': 0, 'exec_pass': 0, 'judge_pass': 0,
                                         'judge_fail': 0, 'exec_fail': 0, 'teacher_error': 0})
        entry['attempts'] += 1
        status = row.get('status')
        if status == 'pass':
            entry['judge_pass'] += 1
            entry['exec_pass'] += 1
        elif status == 'judge_fail':
            entry['judge_fail'] += 1
            entry['exec_pass'] += 1
        elif status == 'teacher_error':
            entry['teacher_error'] += 1
        else:
            entry['exec_fail'] += 1
    accepted_per = {}
    for row in accepted:
        teacher = row.get('teacher_backend', 'unknown')
        accepted_per[teacher] = accepted_per.get(teacher, 0) + 1
    accepted_ids = {row.get('id') for row in accepted if row.get('id')}
    attempt_ids = {row.get('id') for row in attempts if row.get('id')}
    processed = accepted_ids | attempt_ids
    times = [parse_time(row.get('timestamp', '')) for row in attempts]
    times = [t for t in times if t]
    elapsed_h = 0
    rate = 0
    eta = ''
    if times:
        elapsed_h = max(1e-6, (max(times) - min(times)).total_seconds() / 3600)
        rate = len(accepted) / elapsed_h if elapsed_h > 0.02 else 0
        if rate > 0 and len(accepted) < TARGET_TRAJECTORIES:
            eta = f'{(TARGET_TRAJECTORIES - len(accepted)) / rate:.1f} h'
    dataset = dataset_size()
    return {
        'accepted': len(accepted), 'rows': len(accepted) * 2, 'attempts': len(attempts),
        'per_teacher': per, 'accepted_per_teacher': accepted_per,
        'processed_samples': len(processed), 'dataset_rows': dataset['rows'], 'shards': dataset['shards'],
        'elapsed_hours': round(elapsed_h, 2), 'accepted_per_hour': round(rate, 1), 'eta_hours': eta,
        'target_trajectories': TARGET_TRAJECTORIES, 'target_rows': TARGET_ROWS,
    }


def legacy_state():
    rows = []
    for name, path in LEGACY:
        summary = {}
        if (path / 'summary.json').exists():
            try:
                summary = json.loads((path / 'summary.json').read_text())
            except Exception:
                summary = {}
        rows.append({'name': name, 'tasks': summary.get('tasks', 54),
                     'executed': summary.get('execution_passed', '—'),
                     'passed': summary.get('judge_pass', '—'),
                     'rate': summary.get('overall_pass_rate')})
    return rows


def sft_state(base=None, train_path=None):
    base = base or ROOT / 'outputs/sft_magicbrush_27b_oneshot_v1'
    train_path = train_path or ROOT / 'data/sft_magicbrush_oneshot_v1/train_filtered.jsonl'
    log = base / 'train' / 'train.log'
    if not log.exists():
        log = base / 'train.log'
    step = 0
    loss = 0
    mem = 0
    if log.exists():
        for line in log.read_text(errors='ignore').splitlines():
            if line.startswith('{') and '"step"' in line:
                try:
                    row = json.loads(line)
                    step = row.get('step', step)
                    loss = row.get('loss', loss)
                    mem = row.get('peak_memory_gib', mem)
                except Exception:
                    pass
    evals = []
    for ckpt in (100, 200, 300, 400):
        path = base / f'development_{ckpt}' / 'summary.json'
        if path.exists():
            try:
                summary = json.loads(path.read_text())
            except Exception:
                summary = {}
            evals.append({'checkpoint': ckpt, **summary})
            continue
        logs = [base / f'development_{ckpt}.log'] + sorted(base.glob(f'development_{ckpt}_part*.log'))
        logs = [log for log in logs if log.exists()]
        if logs:
            n = parsed = executed = passed = 0
            for log in logs:
                for line in log.read_text(errors='ignore').splitlines():
                    if not line.startswith('{"id"'):
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    n += 1
                    parsed += 1 if row.get('parsed') else 0
                    executed += 1 if row.get('exec') == 'passed' else 0
                    passed += 1 if row.get('pass') is True else 0
            evals.append({'checkpoint': ckpt, 'tasks': n, 'parsed': parsed,
                          'executed': executed, 'judge_pass': passed,
                          'overall_pass_rate': passed / n if n else 0, 'partial': True})
    rows = 0
    if train_path.exists():
        rows = sum(1 for line in train_path.read_text().splitlines() if line.strip())
    zero = {}
    zero_path = ROOT / 'outputs/zero_shot_qwen38_27b_oneshot_dev32/summary.json'
    if zero_path.exists():
        try:
            zero = json.loads(zero_path.read_text())
        except Exception:
            zero = {}
    if not zero:
        zero_log = ROOT / 'outputs/zero_shot_qwen38_27b_oneshot_dev32.log'
        if zero_log.exists():
            n = passed = 0
            for line in zero_log.read_text(errors='ignore').splitlines():
                if not line.startswith('{"id"'):
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                n += 1
                passed += 1 if row.get('pass') is True else 0
            if n:
                zero = {'tasks': n, 'judge_pass': passed,
                        'overall_pass_rate': passed / n if n else 0, 'partial': True}
    return {'step': step, 'total_steps': 400, 'loss': loss, 'peak_memory_gib': mem,
            'train_rows': rows, 'evals': evals, 'zero_shot': zero}


def read_last_step(log):
    step = loss = mem = 0
    if log.exists():
        for line in log.read_text(errors='ignore').splitlines():
            if line.startswith('{') and '"step"' in line:
                try:
                    row = json.loads(line)
                    step = row.get('step', step)
                    loss = row.get('loss', loss)
                    mem = row.get('peak_memory_gib', mem)
                except Exception:
                    pass
    return step, loss, mem


def summary_at(path):
    try:
        return json.loads(Path(path).read_text()) if Path(path).exists() else {}
    except Exception:
        return {}


SFT_RUNS = [
    ('v2 · MagicBrush', 'outputs/sft_magicbrush_27b_oneshot_v1/train', 200, 'MagicBrush 1,195'),
    ('v3 · MagicBrush', 'outputs/sft_magicbrush_27b_oneshot_v3/train', 400, 'MagicBrush 3,306'),
    ('v3-cont · MagicBrush', 'outputs/sft_magicbrush_27b_oneshot_v3_cont/train', 1200, 'MagicBrush 3,306'),
    ('v4 · MagicBrush', 'outputs/sft_magicbrush_27b_v4/train', 1600, 'MagicBrush 5,082'),
    ('mixed v2 · high-ratio', 'outputs/sft_mixed_v2/train', 800, '4,109 (ext 21%)'),
    ('v5 · mixed-all', 'outputs/sft_mixed_v3/train', 2000, '5,962 (ext 15%)'),
    ('v6 · low-ratio', 'outputs/sft_mixed_v6_lowratio/train', 2000, '5,522 (ext 8%)'),
    ('rft · best-of-N', 'outputs/sft_magicbrush_rft_v1/train', 1600, 'RFT 5,250'),
    ('v9 · RFT + target-informed', 'outputs/sft_rft_ti_v1/train', 1600, 'RFT + TI ~5.8k'),
    ('v10 · all-data mixed', 'outputs/sft_all_v10/train', 2000, '7,619 (MB+UE+Sci)'),
    ('v11 · all-data + TI', 'outputs/sft_all_v11/train', 2000, '8,793 (MB+UE+Sci+TI)'),
    ('v12 · 10k all-data', 'outputs/sft_all_v12/train', 2400, '9,447 (RFT+UE+Sci+TI)'),
    ('v13 · all-data + TI v6', 'outputs/sft_all_v13/train', 2400, '9,890 (RFT+UE+Sci+TI1.3k)'),
    ('v13-warm · from best ckpt', 'outputs/sft_v13_warmstart_v2/train', 1500, '9,890 (warm start)'),
    ('v13-r32 · capacity', 'outputs/sft_all_v13_r32_v2/train', 2400, '9,890 (LoRA r32)'),
    ('v15 · TI 2x dose', 'outputs/sft_all_v15_ti2x/train', 2400, '11,209 (TI 23.5%)'),
    ('warm-v12 · no TI control', 'outputs/sft_v12_warm/train', 1500, '9,447 (TI 0%)'),
    ('warm-ti2x · TI 23.5%', 'outputs/sft_v15_warm_ti2x/train', 1500, '11,209 (TI 2x)'),
    ('warm-iter2 · from ckpt-500', 'outputs/sft_warm_iter2/train', 500, '9,890 (2nd iteration)'),
    ('warm-iter3 · 3rd iteration', 'outputs/sft_warm_iter3/train', 500, '9,890 (3rd iteration)'),
    ('v13-long · 4,800 steps', 'outputs/sft_v13_long_cont/train', 4800, '9,890 (long run)'),
    ('v14 · 12k all-data + TI v7', 'outputs/sft_all_v14_cont/train', 2400, '11,978 (TI 26%)'),
    ('v15 · 12k all-data + TI v8', 'outputs/sft_all_v15/train', 2400, '12,034 (TI 29%)'),
    ('v12-fresh · no-TI control', 'outputs/sft_all_v12_fresh/train', 2400, '9,447 (TI 0%)'),
    ('ultraedit-only', 'outputs/sft_ultraedit_27b/train', 800, 'UltraEdit 700'),
]


def partial_summary(log_path):
    """Partial eval metrics from a still-running eval log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return {}
    n = parsed = executed = passed = 0
    for line in log_path.read_text(errors='ignore').splitlines():
        if not line.startswith('{"id"'):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        n += 1
        parsed += 1 if row.get('parsed') else 0
        executed += 1 if row.get('exec') == 'passed' else 0
        passed += 1 if row.get('pass') is True else 0
    if not n:
        return {}
    return {'tasks': n, 'parsed': parsed, 'executed': executed, 'judge_pass': passed,
            'overall_pass_rate': passed / n, 'partial': True}


def experiments_state():
    runs = []
    now = dt.datetime.utcnow().timestamp()
    for name, rel, total, data in SFT_RUNS:
        base = ROOT / rel
        log = base / 'train.log'
        if not log.exists():
            log = base.parent / 'train.log'
        step, loss, mem = read_last_step(log)
        age_min = (now - log.stat().st_mtime) / 60 if log.exists() else None
        if age_min is None:
            status = 'not started'
        elif age_min < 10 and step < total:
            status = 'running'
        elif step >= total:
            status = 'done'
        else:
            status = 'stopped'
        runs.append({'name': name, 'data': data, 'step': step, 'total': total,
                     'loss': loss, 'mem': mem, 'status': status,
                     'age_min': round(age_min, 1) if age_min is not None else None})
    bench = [
        ('zero-shot', 'MagicBrush dev32', 'outputs/zero_shot_qwen38_27b_oneshot_dev32/summary.json'),
        ('zero-shot', 'UltraEdit dev', 'outputs/crossbench_zero_ultraedit/summary.json'),
        ('zero-shot', 'SciForma dev', 'outputs/crossbench_zero_sciforma/summary.json'),
        ('v3-cont-1200', 'MagicBrush dev32', 'outputs/sft_magicbrush_27b_oneshot_v3_cont/development_1200/summary.json'),
        ('v3-cont-1200', 'UltraEdit dev', 'outputs/crossbench_v3cont1200_ultraedit/summary.json'),
        ('v3-cont-1200', 'SciForma dev', 'outputs/crossbench_v3cont1200_sciforma/summary.json'),
        ('mixed v1-200', 'MagicBrush dev32', 'outputs/sft_mixed_oneshot_v1/development_200_magicbrush/summary.json'),
    ]
    for run_name, run_dir in (('v4', 'outputs/sft_magicbrush_27b_v4'),
                              ('v5', 'outputs/sft_mixed_v3'),
                              ('v6', 'outputs/sft_mixed_v6_lowratio'),
                              ('rft', 'outputs/sft_magicbrush_rft_v1'),
                              ('v9', 'outputs/sft_rft_ti_v1')):
        for path in sorted((ROOT / run_dir).glob('development_*/summary.json')):
            ckpt = path.parent.name.replace('development_', '')
            bench.append((f'{run_name}-{ckpt}', 'MagicBrush dev32',
                          str(path.relative_to(ROOT))))
    for ckpt in (200, 400):
        bench.append((f'mixed v2-{ckpt}', 'MagicBrush dev32',
                      f'outputs/sft_mixed_v2/development_{ckpt}_magicbrush/summary.json'))
    rows = []
    for model, bench_name, path in bench:
        summary = summary_at(ROOT / path)
        if not summary:
            summary = partial_summary(ROOT / path.replace('/summary.json', '/results.jsonl'))
        if not summary:
            summary = partial_summary(ROOT / path.replace('/summary.json', '.log'))
        if not summary:
            continue
        rate = summary.get('overall_pass_rate')
        rows.append({'model': model, 'bench': bench_name,
                     'tasks': summary.get('tasks'), 'executed': summary.get('executed'),
                     'pass': summary.get('judge_pass'),
                     'rate': rate if isinstance(rate, (int, float)) else None,
                     'partial': summary.get('partial', False)})
    for tdir in sorted((ROOT / 'data').glob('teacher_eval_*')):
        summary_path = tdir / 'summary.json'
        if not summary_path.exists():
            continue
        try:
            obj = json.loads(summary_path.read_text())
        except Exception:
            continue
        teacher = obj.get('teacher', tdir.name.replace('teacher_eval_', ''))
        for bench_key, label in (('magicbrush', 'MagicBrush dev32'),
                                 ('ultraedit', 'UltraEdit dev'),
                                 ('sciforma', 'SciForma dev')):
            stats = (obj.get('benches') or {}).get(bench_key) or {}
            if not stats.get('tasks') or not stats.get('executed'):
                continue
            rows.append({'model': f'teacher {teacher}', 'bench': label,
                         'tasks': stats.get('tasks'), 'executed': stats.get('executed'),
                         'pass': stats.get('judge_pass'),
                         'rate': stats.get('overall_pass_rate'), 'partial': False})
    return runs, rows


def data_quality_state():
    rft = summary_at(ROOT / 'data/sft_magicbrush_rft_v1/report.json')
    star = summary_at(ROOT / 'data/teacher_target_informed_gpt55/comparison.json').get('summary', {})
    return rft, star


def official_metrics_state():
    rows = []
    bench_labels = {'magicbrush': 'MagicBrush dev32', 'ultraedit': 'UltraEdit dev',
                    'sciforma': 'SciForma dev'}
    for path in sorted((ROOT / 'outputs').glob('*/development_*/official_metrics.json')):
        data = summary_at(path)
        if not data.get('summary'):
            continue
        run = path.parent.parent.name
        ckpt = path.parent.name.replace('development_', '')
        rows.append((f'{run}-{ckpt}', 'MagicBrush dev32', data['summary']))
    for path in sorted((ROOT / 'outputs').glob('crossbench_*/official_metrics.json')):
        data = summary_at(path)
        if not data.get('summary'):
            continue
        name = path.parent.name.replace('crossbench_', '')
        bench = 'MagicBrush dev32'
        for key, label in bench_labels.items():
            if key in name:
                bench = label
        rows.append((name.replace('_', ' '), bench, data['summary']))
    for path in sorted((ROOT / 'data').glob('teacher_eval_*/official_metrics.json')):
        data = summary_at(path)
        if not data.get('summary'):
            continue
        teacher = path.parent.name.replace('teacher_eval_', '')
        rows.append((f'teacher {teacher}', 'MagicBrush dev32', data['summary']))
    return rows


def render(out):
    scale = scale_state()
    legacy = legacy_state()
    sft = sft_state()
    sft_code = sft_state(ROOT / 'outputs/sft_magicbrush_27b_oneshot_code_v1',
                         ROOT / 'data/sft_magicbrush_oneshot_code_v1/train_filtered.jsonl')
    runs, bench_rows = experiments_state()
    rft, star = data_quality_state()
    dq_card = f'''<section class="panel wide">
    <h2>Data quality · RFT &amp; target-informed distillation</h2>
    <table>
      <thead><tr><th>experiment</th><th>n</th><th>judge pass</th><th>CLIP gain vs target</th><th>L1 gain vs target</th><th>notes</th></tr></thead>
      <tbody>
        <tr><td>source-only accepted (base)</td><td>{rft.get('samples', 0):,}</td><td>—</td><td>{rft.get('base_clip_gain_mean', 0):+.4f}</td><td>—</td><td>all accepted trajectories</td></tr>
        <tr><td>RFT best-of-N selected</td><td>{rft.get('selected', 0):,}</td><td>—</td><td>{rft.get('selected_clip_gain_mean', 0):+.4f}</td><td>—</td><td>{rft.get('multi_candidate_samples', 0):,} multi-candidate; {rft.get('selected_clip_gain_positive', 0):,} positive</td></tr>
        <tr><td>STaR target-informed (gpt55)</td><td>{star.get('n', 0)}</td><td>{star.get('star_pass', 0)}</td><td>{star.get('star_clip_gain_mean', 0):+.4f}</td><td>{star.get('star_l1_gain_mean', 0):+.4f}</td><td>flip-reuse {star.get('reuse_passed', 0)}/{star.get('n', 0)}; 0 hardcoded (refined filter)</td></tr>
        <tr><td>source-only matched base</td><td>{star.get('n', 0)}</td><td>—</td><td>{star.get('base_clip_gain_mean', 0):+.4f}</td><td>{star.get('base_l1_gain_mean', 0):+.4f}</td><td>accepted trajectories only</td></tr>
      </tbody>
    </table>
  </section>'''
    official = official_metrics_state()
    om_rows = []
    for model, bench, s in official:
        om_rows.append(f'''<tr><td>{html.escape(model)}</td><td>{html.escape(bench)}</td>
<td>{s.get('n', 0)}</td><td>{s.get('l1', 0):.4f}</td><td>{s.get('psnr', 0):.2f}</td>
<td>{s.get('ssim', 0):.3f}</td><td>{s.get('lpips', 0):.3f}</td>
<td>{s.get('clip_i', 0):.3f}</td><td>{s.get('clip_gain', 0):+.4f}</td><td>{s.get('dino_sim', 0):.3f}</td><td>{s.get('dino_gain', 0):+.4f}</td></tr>''')
    om_table = '\n'.join(om_rows) or '<tr><td colspan="11">waiting for official metrics</td></tr>'
    om_card = f'''<section class="panel wide">
    <h2>Official benchmark metrics (vs target)</h2>
    <table>
      <thead><tr><th>model</th><th>benchmark</th><th>n</th><th>L1 ↓</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>CLIP-I ↑</th><th>CLIP gain ↑</th><th>DINO ↑</th><th>DINO gain ↑</th></tr></thead>
      <tbody>{om_table}</tbody>
    </table>
  </section>'''
    run_rows = []
    for run in runs:
        pct = min(100, 100 * run['step'] / run['total']) if run['total'] else 0
        cls = {'running': 'pass', 'done': 'run', 'stopped': 'warn'}.get(run['status'], 'muted')
        run_rows.append(f'''<tr><td>{html.escape(run['name'])}</td><td>{html.escape(run['data'])}</td>
<td>{run['step']} / {run['total']}</td><td>{run['loss']:.4f}</td>
<td><div class="minibar"><div class="fill" style="width:{pct:.1f}%"></div></div></td>
<td style="color:var(--{cls})">{run['status']}</td></tr>''')
    run_table = '\n'.join(run_rows)
    bench_html = []
    for row in bench_rows:
        rate_text = f"{row['rate']:.1%}" if row['rate'] is not None else '—'
        suffix = ' <span style="color:var(--warn)">partial</span>' if row.get('partial') else ''
        bench_html.append(f'''<tr><td>{html.escape(row['model'])}</td><td>{html.escape(row['bench'])}{suffix}</td>
<td>{row['tasks'] if row['tasks'] is not None else '—'}</td>
<td>{row['executed'] if row['executed'] is not None else '—'}</td>
<td>{row['pass'] if row['pass'] is not None else '—'}</td><td>{rate_text}</td></tr>''')
    bench_table = '\n'.join(bench_html) or '<tr><td colspan="6">waiting for evals</td></tr>'
    experiment_card = f'''<section class="panel wide">
    <h2>Experiment board</h2>
    <table>
      <thead><tr><th>run</th><th>data</th><th>step</th><th>loss</th><th>progress</th><th>status</th></tr></thead>
      <tbody>{run_table}</tbody>
    </table>
    <table style="margin-top:16px">
      <thead><tr><th>model</th><th>benchmark</th><th>tasks</th><th>executed</th><th>judge pass</th><th>pass rate</th></tr></thead>
      <tbody>{bench_table}</tbody>
    </table>
  </section>'''
    now = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    traj_pct = min(100, 100 * scale['accepted'] / TARGET_TRAJECTORIES)
    row_pct = min(100, 100 * scale['rows'] / TARGET_ROWS)
    coverage_pct = min(100, 100 * scale['processed_samples'] / scale['dataset_rows']) if scale['dataset_rows'] else 0
    teacher_rows = []
    for teacher in sorted(scale['per_teacher'], key=lambda k: -scale['accepted_per_teacher'].get(k, 0)):
        entry = scale['per_teacher'][teacher]
        accepted = scale['accepted_per_teacher'].get(teacher, 0)
        rate = accepted / entry['attempts'] if entry['attempts'] else 0
        teacher_rows.append(f'''<tr><td>{html.escape(teacher)}</td><td>{entry['attempts']}</td>
<td>{entry['exec_pass']}</td><td>{entry['judge_pass']}</td><td>{accepted}</td>
<td>{rate:.1%}</td><td>{entry['exec_fail']}</td><td>{entry['teacher_error']}</td></tr>''')
    teacher_table = '\n'.join(teacher_rows) or '<tr><td colspan="8">no attempts yet</td></tr>'
    recent = load_jsonl(SCALE / 'judged.jsonl')[-10:][::-1]
    recent_rows = []
    for row in recent:
        judge = row.get('judge') or {}
        recent_rows.append(f'''<tr><td class="mono">{html.escape(row['id'])}</td>
<td>{html.escape(row.get('instruction', ''))}</td><td>{html.escape(row.get('teacher_backend', ''))}</td>
<td>{judge.get('instruction_adherence', '—')}/{judge.get('preservation', '—')}/{judge.get('visual_quality', '—')}</td></tr>''')
    recent_table = '\n'.join(recent_rows) or '<tr><td colspan="4">no accepted samples yet</td></tr>'
    legacy_rows = []
    for row in legacy:
        rate = row['rate']
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else '—'
        legacy_rows.append(f'''<tr><td>{html.escape(row['name'])}</td><td>{row['tasks']}</td>
<td>{row['executed']}</td><td>{row['passed']}</td><td>{rate_text}</td></tr>''')
    legacy_table = '\n'.join(legacy_rows)
    sft_pct = min(100, 100 * sft['step'] / sft['total_steps']) if sft['total_steps'] else 0
    sft_eval_rows = []
    for entry in sft['evals']:
        rate = entry.get('overall_pass_rate')
        rate_text = f'{rate:.1%}' if isinstance(rate, (int, float)) else '—'
        sft_eval_rows.append(f'''<tr><td>{entry['checkpoint']}</td><td>{entry.get('tasks', '—')}</td>
<td>{entry.get('parsed', '—')}</td><td>{entry.get('executed', '—')}</td>
<td>{entry.get('judge_pass', '—')}</td><td>{rate_text}</td></tr>''')
    sft_eval_table = '\n'.join(sft_eval_rows) or '<tr><td colspan="6">waiting for first checkpoint</td></tr>'
    if sft.get('zero_shot'):
        zero = sft['zero_shot']
        zero_text = f"{zero.get('judge_pass', 0)}/{zero.get('tasks', 32)} = {zero.get('overall_pass_rate', 0):.1%}"
        zero_label = 'zero-shot baseline (dev32)'
    else:
        zero_text = '1/8 = 12.5%'
        zero_label = 'zero-shot baseline (dev8)'
    sft_card = f'''<section class="panel wide">
    <h2>SFT · Qwen3.8-27B one-shot</h2>
    <div class="bigbar"><div class="fill" style="width:{sft_pct:.2f}%"></div></div>
    <div class="stats">
      <div class="stat"><b>{sft['step']} / {sft['total_steps']}</b><span>training step</span></div>
      <div class="stat"><b>{sft['loss']:.4f}</b><span>last loss</span></div>
      <div class="stat"><b>{sft['peak_memory_gib']:.1f} GB</b><span>peak memory</span></div>
      <div class="stat"><b>{sft['train_rows']:,}</b><span>train rows</span></div>
      <div class="stat"><b>{zero_text}</b><span>{zero_label}</span></div>
    </div>
    <table style="margin-top:12px">
      <thead><tr><th>checkpoint</th><th>dev tasks</th><th>parsed</th><th>executed</th><th>judge pass</th><th>pass rate</th></tr></thead>
      <tbody>{sft_eval_table}</tbody>
    </table>
  </section>'''
    sft_code_pct = min(100, 100 * sft_code['step'] / sft_code['total_steps']) if sft_code['total_steps'] else 0
    sft_code_eval_rows = []
    for entry in sft_code['evals']:
        rate = entry.get('overall_pass_rate')
        rate_text = f'{rate:.1%}' if isinstance(rate, (int, float)) else '—'
        sft_code_eval_rows.append(f'''<tr><td>{entry['checkpoint']}</td><td>{entry.get('tasks', '—')}</td>
<td>{entry.get('parsed', '—')}</td><td>{entry.get('executed', '—')}</td>
<td>{entry.get('judge_pass', '—')}</td><td>{rate_text}</td></tr>''')
    sft_code_table = '\n'.join(sft_code_eval_rows) or '<tr><td colspan="6">waiting for checkpoint-200</td></tr>'
    sft_code_card = f'''<section class="panel wide">
    <h2>SFT · Qwen3.8-27B one-shot (code-only)</h2>
    <div class="bigbar"><div class="fill" style="width:{sft_code_pct:.2f}%"></div></div>
    <div class="stats">
      <div class="stat"><b>{sft_code['step']} / {sft_code['total_steps']}</b><span>training step</span></div>
      <div class="stat"><b>{sft_code['loss']:.4f}</b><span>last loss</span></div>
      <div class="stat"><b>{sft_code['peak_memory_gib']:.1f} GB</b><span>peak memory</span></div>
      <div class="stat"><b>{sft_code['train_rows']:,}</b><span>train rows</span></div>
      <div class="stat"><b>{zero_text}</b><span>{zero_label}</span></div>
    </div>
    <table style="margin-top:12px">
      <thead><tr><th>checkpoint</th><th>dev tasks</th><th>parsed</th><th>executed</th><th>judge pass</th><th>pass rate</th></tr></thead>
      <tbody>{sft_code_table}</tbody>
    </table>
  </section>'''
    page = f'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Qwen Pixel-Edit · SFT &amp; RL dashboard</title>
<style>
:root{{--bg:#0b1020;--panel:#141a2e;--line:#26304a;--text:#e8ecf7;--muted:#9aa7c2;--pass:#22c55e;--fail:#ef4444;--warn:#f59e0b;--run:#60a5fa;--accent:#a855f7}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}}
header{{max-width:1200px;margin:0 auto;padding:34px 24px 8px}}
h1{{margin:0 0 4px;font-size:28px}}
.sub{{color:var(--muted);margin:0 0 18px}}
.hero{{background:linear-gradient(135deg,#171f38,#12172a);border:1px solid var(--line);border-radius:20px;padding:22px 24px;margin:18px 0}}
.hero-line{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}}
.hero-num{{font-size:44px;font-weight:800;letter-spacing:-1px}}
.hero-num span{{font-size:20px;color:var(--muted);font-weight:600}}
.hero-label{{color:var(--muted);font-size:13px}}
.bigbar{{height:16px;background:#0a0f1d;border:1px solid var(--line);border-radius:999px;overflow:hidden;margin:10px 0 18px}}
.bigbar .fill{{height:100%;background:linear-gradient(90deg,#22c55e,#16a34a);transition:width .5s}}
.bigbar.rows .fill{{background:linear-gradient(90deg,#a855f7,#6366f1)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:8px}}
.stat{{background:#0f1526;border:1px solid var(--line);border-radius:12px;padding:10px 12px}}
.stat b{{display:block;font-size:22px}}
.stat span{{color:var(--muted);font-size:11px}}
main{{max-width:1200px;margin:0 auto;padding:0 24px 40px;display:grid;grid-template-columns:1.4fr 1fr;gap:18px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}}
.panel.wide{{grid-column:1/-1}}
h2{{margin:0 0 10px;font-size:17px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
tr:last-child td{{border-bottom:none}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}}
.coverage{{color:var(--muted);font-size:12px;margin-top:8px}}
footer{{max-width:1200px;margin:0 auto;padding:0 24px 44px;color:var(--muted);font-size:12px}}
details{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 16px;margin-top:14px}}
summary{{cursor:pointer;color:var(--run)}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <h1>Qwen Pixel-Edit · SFT &amp; RL dashboard</h1>
  <p class="sub">Instruction-based image editing with executable pixel programs. Teacher-distilled, execution-verified SFT data, then RL. Auto-refresh every 30s.</p>
  <div class="hero">
    <div class="hero-line">
      <div class="hero-num">{scale['accepted']:,} <span>/ {TARGET_TRAJECTORIES:,}</span></div>
      <div class="hero-label">accepted trajectories</div>
    </div>
    <div class="bigbar"><div class="fill" style="width:{traj_pct:.2f}%"></div></div>
    <div class="hero-line">
      <div class="hero-num" style="font-size:30px">{scale['rows']:,} <span>/ {TARGET_ROWS:,}</span></div>
      <div class="hero-label">estimated SFT rows</div>
    </div>
    <div class="bigbar rows"><div class="fill" style="width:{row_pct:.2f}%"></div></div>
    <div class="stats">
      <div class="stat"><b>{scale['attempts']:,}</b><span>teacher attempts</span></div>
      <div class="stat"><b>{scale['accepted_per_hour']:,}</b><span>accepted / hour</span></div>
      <div class="stat"><b>{scale['eta_hours'] or '—'}</b><span>ETA to 5k</span></div>
      <div class="stat"><b>{scale['processed_samples']:,} / {scale['dataset_rows']:,}</b><span>samples touched</span></div>
      <div class="stat"><b>{coverage_pct:.1f}%</b><span>dataset coverage</span></div>
      <div class="stat"><b>{now}</b><span>last render</span></div>
    </div>
  </div>
</header>
<main>
  {experiment_card}
  {dq_card}
  {om_card}
  {sft_card}
  {sft_code_card}
  <section class="panel">
    <h2>Per-teacher contribution</h2>
    <table>
      <thead><tr><th>teacher</th><th>attempts</th><th>exec ok</th><th>judge pass</th><th>accepted</th><th>accept rate</th><th>exec fail</th><th>teacher error</th></tr></thead>
      <tbody>{teacher_table}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Recent accepted</h2>
    <table>
      <thead><tr><th>sample</th><th>instruction</th><th>teacher</th><th>scores</th></tr></thead>
      <tbody>{recent_table}</tbody>
    </table>
  </section>
  <section class="panel wide">
    <h2>Dataset coverage</h2>
    <p class="coverage">Touched {scale['processed_samples']:,} of {scale['dataset_rows']:,} MagicBrush train rows across {scale['shards']} shards ({coverage_pct:.1f}%). Accepted {scale['accepted']:,} trajectories; target {TARGET_TRAJECTORIES:,} trajectories / {TARGET_ROWS:,} SFT rows.</p>
  </section>
</main>
<footer>
  <details>
    <summary>Completed 54-sample teacher comparison</summary>
    <table>
      <thead><tr><th>teacher</th><th>tasks</th><th>executed</th><th>judge pass</th><th>pass rate</th></tr></thead>
      <tbody>{legacy_table}</tbody>
    </table>
  </details>
</footer>
</body>
</html>'''
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_text(page)
    (out / 'progress.json').write_text(json.dumps({'rendered_at_utc': now, 'scale': scale,
                                                   'sft': sft, 'sft_code': sft_code,
                                                   'legacy': legacy}, indent=2))
    print(json.dumps({'rendered_at_utc': now, 'scale': scale}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'outputs/magicbrush_teacher_progress'))
    args = parser.parse_args()
    render(Path(args.out))


if __name__ == '__main__':
    main()
