"""Create a standalone page with actual Qwen actions and edited pixels."""
import base64
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/agent_demo'


def image_tag(path):
    path = Path(path)
    if not path.exists():
        return '<p>尚未产生输出</p>'
    encoded = base64.b64encode(path.read_bytes()).decode()
    return '<img src="data:image/png;base64,' + encoded + '">'


def main():
    rollout_dir = OUT / 'qwen_refine' if (OUT / 'qwen_refine/summary.json').exists() else OUT / 'qwen_sft'
    refined = rollout_dir.name == 'qwen_refine'
    trajectory = json.loads((rollout_dir / 'trajectory.json').read_text())
    summary_path = rollout_dir / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        'complete': False, 'running': True, 'steps': len(trajectory['turns']),
        'actions': [t['action'] for t in trajectory['turns']]}
    checks = json.loads((ROOT / 'data/agent_demo_v1/encoding_check.json').read_text())
    gradients = json.loads((OUT / ('sft_refine' if refined else 'sft') / 'gradient_check.json').read_text())
    turns = []
    for turn in trajectory['turns']:
        name = turn['action']
        content = turn.get('arguments', {}).get('code', turn.get('generation', ''))
        feedback = turn.get('feedback', {}).get('content', '')
        output = '<pre>' + html.escape(str(feedback)) + '</pre>'
        if isinstance(feedback, list):
            output = ''.join(image_tag(b['image']) if b.get('type') == 'image'
                             else '<p>' + html.escape(b['text']) + '</p>' for b in feedback)
        turns.append('<details open><summary>步骤 ' + str(turn['step'] + 1)
                     + ' · ' + html.escape(name) + ' · ' + str(round(turn.get('seconds', 0), 1))
                     + 's</summary><pre>' + html.escape(content) + '</pre>' + output + '</details>')
    status = '已提交匹配的 PNG 和 RGB 像素数组' if summary['complete'] else '尚未完成有效提交'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen Pixel · Agent Demo</title><style>
body{font-family:system-ui,sans-serif;background:#f5f6f8;color:#18202c;max-width:1120px;margin:auto;padding:24px}
h1{margin-bottom:8px}p{line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
article,details{background:white;border:1px solid #dce1e7;border-radius:12px;padding:16px;margin:16px 0}
img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f6;padding:14px;font-size:13px}
summary{cursor:pointer;font-weight:600}.note{color:#58677a}@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style><h1>Qwen Pixel · Agent Demo</h1>
<p>任务：将两辆公交车的红色车身改成蓝色，保留车窗、轮胎、道路和背景。</p>
<p class="note">单任务训练流程验证：1 条实际执行示范、5 个监督操作、''' + ('150' if refined else '30') + ''' 步 LoRA SFT。下面是 Qwen 自己选择工具并执行的真实轨迹；同一训练任务上的展示不能说明泛化能力。</p>
<div class="grid"><article><h3>原图</h3>''' + image_tag(rollout_dir / 'source.png') + '''</article>
<article><h3>教师示范</h3>''' + image_tag(OUT / 'teacher/edited.png') + '''</article>
<article><h3>Qwen SFT 输出</h3>''' + image_tag(rollout_dir / 'edited.png') + '''</article></div>
<article><h3>执行验证</h3><p>''' + status + '</p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '''</pre></article>
<h2>Qwen 实际操作</h2>''' + ''.join(turns) + '''<details><summary>SFT 编码和梯度检查</summary><pre>''' + html.escape(json.dumps({'encoding': checks, 'gradients': gradients}, ensure_ascii=False, indent=2)) + '</pre></details></html>'
    (OUT / 'index.html').write_text(page)
    if (rollout_dir / 'edited.png').exists():
        source = Image.open(rollout_dir / 'source.png').convert('RGB')
        edited = Image.open(rollout_dir / 'edited.png').convert('RGB')
        w, h = source.size
        comparison = Image.new('RGB', (w * 2, h + 28), 'white')
        comparison.paste(source, (0, 28)); comparison.paste(edited, (w, 28))
        draw = ImageDraw.Draw(comparison)
        draw.text((12, 8), 'SOURCE', fill='black'); draw.text((w + 12, 8), 'QWEN AGENT SFT', fill='black')
        comparison.save(OUT / 'comparison.png')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
