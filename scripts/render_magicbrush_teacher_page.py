"""Render the GPT-6.1-Sol MagicBrush teacher audit as a self-contained HTML page."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan/qwen_pixel')


def category(instruction):
    text = instruction.lower()
    if any(k in text for k in ('add ', 'put a', 'put an', 'give ', 'have the', 'wear')):
        return 'add / insert'
    if any(k in text for k in ('remove', 'leave only', 'delete', 'without ')):
        return 'remove'
    if any(k in text for k in ('replace', 'change', 'turn ', 'make ', 'let ', 'what if')):
        return 'replace / change'
    return 'other'


def thumb(src, dst, size=384):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert('RGB')
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        image.save(dst, format='JPEG', quality=86, optimize=True)


def badge(pass_value, label):
    cls = 'pass' if pass_value else 'fail'
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judged', default=str(ROOT / 'data/teacher_magicbrush_gpt61_inpaint_54_v1/judged.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'outputs/magicbrush_gpt61_teacher_report'))
    args = parser.parse_args()
    judged = Path(args.judged)
    out = Path(args.out)
    assets = out / 'assets'
    if out.exists():
        shutil.rmtree(out)
    assets.mkdir(parents=True)
    samples = [json.loads(line) for line in judged.read_text().splitlines() if line.strip()]
    executed = [s for s in samples if s['execution']['status'] == 'passed']
    passed = [s for s in samples if (s.get('judge') or {}).get('pass')]
    cards = []
    categories = {}
    for sample in samples:
        sid = sample['id']
        judge = sample.get('judge') or {}
        ok = bool(judge.get('pass'))
        status = sample['execution']['status']
        cat = category(sample['instruction'])
        categories[cat] = categories.get(cat, 0) + 1
        source_rel = edited_rel = target_rel = None
        try:
            source_rel = f'assets/{sid}_source.jpg'
            thumb(sample['source_image'], out / source_rel)
        except Exception:
            source_rel = None
        try:
            target_rel = f'assets/{sid}_target.jpg'
            thumb(sample['target_image'], out / target_rel)
        except Exception:
            target_rel = None
        if status == 'passed':
            try:
                edited_rel = f'assets/{sid}_edited.jpg'
                thumb(sample['execution']['output'], out / edited_rel)
            except Exception:
                edited_rel = None
        scores = ''
        if judge.get('instruction_adherence') is not None:
            scores = (f'<span class="score">instr {judge.get("instruction_adherence")}</span>'
                      f'<span class="score">preserve {judge.get("preservation")}</span>'
                      f'<span class="score">quality {judge.get("visual_quality")}</span>')
        reason = html.escape(str(judge.get('reason', '')))
        metrics = sample.get('metrics') or {}
        metric_text = ''
        if metrics:
            metric_text = ' · '.join(f'{k}={metrics[k]:.4f}' for k in ('l1', 'gain', 'mask_l1', 'preserve_l1') if isinstance(metrics.get(k), (int, float)))
        code = html.escape(sample.get('code', '') or '(empty)')
        reasoning = html.escape(json.dumps(sample.get('reasoning') or {}, indent=1, ensure_ascii=False))
        image_block = '<div class="images">'
        if source_rel:
            image_block += f'<figure><img src="{source_rel}" loading="lazy"><figcaption>source</figcaption></figure>'
        if edited_rel:
            image_block += f'<figure><img src="{edited_rel}" loading="lazy"><figcaption>GPT-6.1-Sol output</figcaption></figure>'
        else:
            image_block += '<figure class="missing"><div>execution failed</div><figcaption>no output</figcaption></figure>'
        if target_rel:
            image_block += f'<figure><img src="{target_rel}" loading="lazy"><figcaption>MagicBrush target (reference only)</figcaption></figure>'
        image_block += '</div>'
        cards.append(f'''
<article class="card" data-status="{'pass' if ok else ('fail' if status == 'passed' else 'error')}" data-category="{html.escape(cat)}">
  <div class="card-head">
    <div>
      <div class="instruction">{html.escape(sample['instruction'])}</div>
      <div class="meta">{html.escape(sid)} · {html.escape(cat)} · execution={html.escape(status)}</div>
    </div>
    <div class="badges">{badge(ok, 'PASS' if ok else ('FAIL' if status == 'passed' else 'EXEC FAIL'))}{scores}</div>
  </div>
  {image_block}
  <p class="reason">{reason}</p>
  <p class="metrics">{html.escape(metric_text)}</p>
  <details><summary>reasoning / plan</summary><pre>{reasoning}</pre></details>
  <details><summary>pixel code</summary><pre>{code}</pre></details>
</article>''')
    cards_html = '\n'.join(cards)
    summary = {
        'samples': len(samples), 'executed': len(executed), 'pass': len(passed),
        'overall': len(passed) / len(samples) if samples else 0,
        'executed_rate': len(passed) / len(executed) if executed else 0,
        'categories': categories,
    }
    page = f'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT-6.1-Sol · MagicBrush inpaint teacher audit</title>
<style>
:root{{--bg:#0b1020;--panel:#141a2e;--line:#26304a;--text:#e8ecf7;--muted:#9aa7c2;--pass:#22c55e;--fail:#ef4444;--warn:#f59e0b;--accent:#60a5fa}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}}
header{{padding:32px 24px 16px;max-width:1400px;margin:0 auto}}
h1{{margin:0 0 6px;font-size:26px}}
.sub{{color:var(--muted);margin:0 0 18px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}}
.stat .v{{font-size:26px;font-weight:700}}
.stat .k{{color:var(--muted);font-size:12px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 24px}}
button{{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:999px;padding:8px 14px;cursor:pointer}}
button.active{{border-color:var(--accent);color:var(--accent)}}
main{{max-width:1400px;margin:0 auto;padding:0 24px 48px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;overflow:hidden}}
.card-head{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}
.instruction{{font-size:16px;font-weight:650;margin-bottom:4px}}
.meta{{color:var(--muted);font-size:12px}}
.badges{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}
.badge{{font-size:11px;font-weight:700;padding:4px 8px;border-radius:999px;border:1px solid}}
.badge.pass{{color:var(--pass);border-color:var(--pass);background:#052e16}}
.badge.fail{{color:var(--fail);border-color:var(--fail);background:#2a0a0a}}
.score{{font-size:11px;color:var(--muted);border:1px solid var(--line);padding:4px 7px;border-radius:999px}}
.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}}
figure{{margin:0;background:#0a0f1d;border:1px solid var(--line);border-radius:10px;overflow:hidden}}
figure img{{width:100%;display:block;aspect-ratio:1/1;object-fit:cover}}
figcaption{{font-size:11px;color:var(--muted);padding:6px 8px;text-align:center}}
figure.missing div{{aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;color:var(--fail);font-weight:700}}
.reason{{color:#cdd6ea;margin:10px 0 4px}}
.metrics{{color:var(--muted);font-size:12px;margin:0 0 10px;font-variant-numeric:tabular-nums}}
details{{border-top:1px solid var(--line);padding-top:8px;margin-top:8px}}
summary{{cursor:pointer;color:var(--accent);font-size:12px}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:11px;color:#c7d2fe;max-height:320px;overflow:auto;background:#0a0f1d;padding:10px;border-radius:8px}}
footer{{max-width:1400px;margin:0 auto;padding:0 24px 40px;color:var(--muted);font-size:12px}}
</style>
</head>
<body>
<header>
  <h1>GPT-6.1-Sol · MagicBrush inpaint teacher audit</h1>
  <p class="sub">54 MagicBrush train samples previously flagged <code>requires_inpaint</code>. The teacher must write pixel code directly (cv2.inpaint / patch synthesis / seamlessClone), execute it, and produce the RGB result. Judge is GPT-5.5; pass = instruction ≥4, preservation ≥4, quality ≥3. Not the official MagicBrush metric.</p>
  <div class="stats">
    <div class="stat"><div class="v">{summary['samples']}</div><div class="k">samples</div></div>
    <div class="stat"><div class="v">{summary['executed']}</div><div class="k">executed</div></div>
    <div class="stat"><div class="v">{summary['pass']}</div><div class="k">judge pass</div></div>
    <div class="stat"><div class="v">{summary['overall']:.1%}</div><div class="k">overall pass rate</div></div>
    <div class="stat"><div class="v">{summary['executed_rate']:.1%}</div><div class="k">of executed</div></div>
  </div>
  <div class="controls">
    <button class="active" data-filter="all">all</button>
    <button data-filter="pass">pass</button>
    <button data-filter="fail">fail</button>
    <button data-filter="error">execution failed</button>
  </div>
</header>
<main>
{cards_html}
</main>
<footer>MagicBrush target images are shown as references only; they were never given to the teacher or the judge. Metrics vs target are diagnostic, not the pass criterion.</footer>
<script>
const buttons=[...document.querySelectorAll('button[data-filter]')];
const cards=[...document.querySelectorAll('.card')];
buttons.forEach(b=>b.addEventListener('click',()=>{{
  buttons.forEach(x=>x.classList.remove('active'));b.classList.add('active');
  const f=b.dataset.filter;
  cards.forEach(c=>c.style.display=(f==='all'||c.dataset.status===f)?'':'none');
}}));
</script>
</body>
</html>'''
    (out / 'index.html').write_text(page)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'out': str(out / 'index.html'), **summary}, indent=2))


if __name__ == '__main__':
    main()
