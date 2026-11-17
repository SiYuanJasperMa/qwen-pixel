"""Build an immediate visual preview of Qwen Pixel generations."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pixel.palette import parse_palette_grid, render_palette_grid  # noqa: E402

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
except OSError:
    FONT = ImageFont.load_default()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def panel(image, size=192):
    image = image.convert('RGB')
    canvas = Image.new('RGB', (size, size), 'white')
    image.thumbnail((size, size), Image.Resampling.NEAREST)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def label(image, text, color='black'):
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 18], fill='white')
    draw.text((3, 2), text[:60], fill=color, font=FONT)
    return image


def b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/eval_smoke_16_4.jsonl')
    parser.add_argument('--predictions', default='outputs/gen_smoke_16_300.jsonl')
    parser.add_argument('--eval-dir', default='outputs/eval_smoke_16_300')
    parser.add_argument('--out', default='outputs/preview_16_300')
    parser.add_argument('--title', default='Qwen Pixel 16x16 smoke')
    args = parser.parse_args()

    rows = {r['id']: r for r in read_jsonl(args.data)}
    preds = {r['id']: r for r in read_jsonl(args.predictions)}
    metrics = {}
    eval_path = resolve(args.eval_dir) / 'per_case.jsonl'
    if eval_path.exists():
        metrics = {r['id']: r for r in read_jsonl(eval_path)}
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cards = []
    sheet_rows = []
    for case_id, row in rows.items():
        pred = preds.get(case_id, {'text': ''})
        task = row['task']
        source_img = None
        if row.get('image'):
            source_img = panel(Image.open(resolve(row['image'])))
        target_img = None
        try:
            tp, tg = parse_palette_grid(row['answer'])
            target_img = panel(render_palette_grid(tp, tg, scale=16))
        except Exception:
            pass
        pred_img = None
        try:
            pp, pg = parse_palette_grid(pred['text'])
            pred_img = panel(render_palette_grid(pp, pg, scale=16))
        except Exception:
            pred_img = Image.new('RGB', (192, 192), 'white')
            ImageDraw.Draw(pred_img).text((8, 90), 'PARSE FAIL', fill='red')
        m = metrics.get(case_id, {})
        caption = f"{task} {case_id}"
        panels = []
        if source_img is not None:
            panels.append(('Input', source_img))
        if target_img is not None:
            panels.append(('Target', target_img))
        panels.append(('Prediction', pred_img))
        if task in {'t2p', 'i2p', 'edit'} and len(panels) >= 2:
            sheet_rows.append((caption, panels, m))
        encoded = [(name, b64(img)) for name, img in panels]
        cards.append({
            'id': case_id, 'task': task, 'instruction': row['instruction'],
            'answer': row['answer'], 'text': pred.get('text', ''), 'metrics': m,
            'panels': encoded,
        })

    # Contact sheet: up to 8 cases, one row each.
    if sheet_rows:
        selected = sheet_rows[:8]
        row_h = 260
        width = 192 * 3
        height = row_h * len(selected)
        sheet = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (caption, panels, m) in enumerate(selected):
            y = i * row_h
            draw.text((3, y + 2), caption[:90], fill='black', font=FONT)
            if len(panels) == 2:
                placeholder = Image.new('RGB', (192, 192), 'white')
                pd = ImageDraw.Draw(placeholder)
                pd.text((8, 84), 'text prompt only', fill='#888888', font=FONT)
                panels = [('Text prompt', placeholder)] + panels
            x = 0
            for name, img in panels[:3]:
                sheet.paste(label(img.copy(), name), (x, y + 22))
                x += 192
            extra = ''
            if 'pixel_accuracy' in m:
                extra = f" pixel_acc={m['pixel_accuracy']:.3f}"
            draw.text((3, y + 238), extra, fill='black', font=FONT)
        sheet.save(out / 'gallery.png')

    cards_html = []
    for card in cards:
        images = ''.join(
            f'<figure><img src="data:image/png;base64,{data}"><figcaption>{html.escape(name)}</figcaption></figure>'
            for name, data in card['panels'])
        m = card['metrics']
        detail = ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in m.items() if k in {'pixel_accuracy', 'changed_accuracy',
                                                          'preserve_accuracy', 'no_op_gain', 'exact', 'parsed'})
        cards_html.append(f'''<article class="card" data-task="{card['task']}">
<p><span class="pill">{card['task']}</span> <code>{html.escape(card['id'])}</code></p>
<p class="instr">{html.escape(card['instruction'][:300])}</p>
<div class="imgs">{images}</div>
<p class="metric">{html.escape(detail)}</p>
<details><summary>raw prediction</summary><pre>{html.escape(card['text'])}</pre></details>
</article>''')

    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
body{{margin:0;background:#0b0f14;color:#edf2f7;font:14px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px}}
h1{{font-size:30px;margin:0 0 6px}}h2{{margin:28px 0 10px}}
.note{{color:#a0adbc}}code{{font-size:12px}}
.filters button{{background:#1b2531;color:#a0adbc;border:1px solid #2b3745;border-radius:8px;padding:6px 10px;margin-right:6px;cursor:pointer}}
.filters button.on{{background:#b8ed94;color:#17220f}}
.card{{background:#141b24;border:1px solid #2b3745;border-radius:12px;padding:16px;margin:12px 0}}
.pill{{background:#1e3323;color:#b8ed94;border:1px solid #3d5c45;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700}}
.instr{{color:#a0adbc}}.imgs{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0}}figure img{{width:192px;height:192px;image-rendering:pixelated;border:1px solid #2b3745;border-radius:6px;background:#fff}}
figcaption{{color:#a0adbc;font-size:11px;text-align:center}}
.metric{{color:#94ccff;font-size:12px}}pre{{overflow:auto;background:#0d1319;padding:10px;border-radius:6px;font-size:11px}}
.gallery{{width:100%;max-width:576px;border:1px solid #2b3745;border-radius:8px;background:#fff}}
</style>
<main><h1>{html.escape(args.title)}</h1>
<p class="note">Checkpoint-300, 16x16 synthetic held-out. 当前模型已学会 PALETTE 和格式，但 GRID 仍以背景色为主；这是训练中间态，不是最终结果。</p>
<h2>Contact sheet</h2>
<img class="gallery" src="gallery.png">
<h2>Cases</h2>
<div class="filters"><button class="on" data-f="all">all</button><button data-f="t2p">t2p</button><button data-f="i2p">i2p</button><button data-f="edit">edit</button><button data-f="pu">pu</button></div>
{''.join(cards_html)}
</main>
<script>
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>c.style.display=(b.dataset.f==='all'||c.dataset.task===b.dataset.f)?'':'none');
}});
</script></body></html>'''
    (out / 'index.html').write_text(page)
    print(out / 'index.html')
    print(out / 'gallery.png')


if __name__ == '__main__':
    main()
