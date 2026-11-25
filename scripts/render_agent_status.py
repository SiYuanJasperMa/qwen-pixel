"""Standalone, shareable status page from persisted real experimental evidence."""
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/agent_status'


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def picture(path):
    path = Path(path)
    if not path.exists():
        return '<p>尚未产生图片</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return '<img src="data:image/png;base64,' + data + '">'


def render_case(folder, title):
    folder = Path(folder)
    trajectory = load(folder / 'trajectory.json')
    if not trajectory:
        return ''
    instruction = trajectory.get('instruction', trajectory.get('task', {}).get('instruction', ''))
    result = load(folder / 'evaluation.json') or load(folder / 'result.json') or load(folder / 'summary.json', {})
    verdict = result.get('judge', {}).get('pass')
    status = '判图通过' if verdict else ('判图未通过' if verdict is False else '等待质量评估')
    details = []
    for turn in trajectory.get('turns', []):
        action = turn.get('action', '')
        code = turn.get('arguments', {}).get('code')
        details.append('<li>' + html.escape(action) + ' · ' + ('执行通过' if turn.get('passed') else '失败或未提交')
                       + ('<pre>' + html.escape(code) + '</pre>' if code else '') + '</li>')
    return '<article><h3>' + html.escape(title) + '</h3><p>' + html.escape(instruction) + '</p><p>' + status + '</p><div class="pair"><div><p>输入</p>' + picture(folder / 'source.png') + '</div><div><p>模型输出</p>' + picture(folder / 'edited.png') + '</div></div><details><summary>实际工具轨迹</summary><ol>' + ''.join(details) + '</ol></details></article>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    teacher_root = ROOT / 'data/agent_teacher_scale_v2'
    teacher_results = [load(p) for p in teacher_root.glob('coco_agent_*/result.json')]
    teacher_count = len(teacher_results)
    teacher_pass = sum(r.get('judge', {}).get('pass', False) for r in teacher_results)
    teacher_valid = sum(r.get('complete', False) for r in teacher_results)
    evaluations = {}
    for name, path in [('固定快照 Instruct · 20 张留出图', 'agent_snapshot_v3/eval_base'),
                       ('固定快照 SFT · 20 张留出图', 'agent_snapshot_v3/eval_sft'),
                       ('固定快照 RL · 20 张留出图', 'agent_snapshot_v3/eval_rl'),
                       ('原生 RGB Instruct · 20 张留出图', 'agent_rgb_v6/base'),
                       ('原生 RGB SFT · 20 张留出图', 'agent_rgb_v6/eval_sft_rgb'),
                       ('保守 LoRA · 留出图', 'agent_stable_v10/eval')]:
        directory = ROOT / 'outputs' / path
        results = [load(p) for p in directory.glob('coco_agent_*/evaluation.json')]
        evaluations[name] = {'已评测': len(results), '有效提交': sum(r['complete'] for r in results),
                            '判图通过': sum(r.get('judge', {}).get('pass', False) for r in results)}
    rl = load(ROOT / 'outputs/agent_snapshot_v3/grpo/summary.json')
    if not rl:
        metrics = ROOT / 'outputs/agent_snapshot_v3/grpo/metrics.jsonl'
        rows = [json.loads(l) for l in metrics.open()] if metrics.exists() else []
        rl = {'completed_groups': len(rows), 'verified_updates': sum(r.get('updated', False) for r in rows)}
    table = ''.join('<tr><td>' + html.escape(name) + '</td>'
                    + ''.join('<td>' + str(v) + '</td>' for v in result.values()) + '</tr>' for name, result in evaluations.items())
    examples = []
    refined = ROOT / 'outputs/agent_demo/qwen_refine'
    examples.append(render_case(refined if (refined / 'summary.json').exists() else ROOT / 'outputs/agent_demo/qwen_sft', '公交车单任务 SFT demo（训练内样本）'))
    for folder in sorted((ROOT / 'outputs/agent_snapshot_v3/eval_sft').glob('coco_agent_*'))[:3]:
        examples.append(render_case(folder, '固定快照 SFT 留出任务 · ' + folder.name))
    for folder in sorted((ROOT / 'outputs/agent_rgb_v6/eval_sft_rgb').glob('coco_agent_*'))[:3]:
        examples.append(render_case(folder, '原生 RGB SFT 留出任务 · ' + folder.name))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen Pixel · Agent SFT / RL</title><style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:28px;background:#f5f7fa;color:#172333}
h1{font-size:32px;margin-bottom:8px}p{line-height:1.7}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
article,.card{background:white;border:1px solid #dce3ec;border-radius:14px;padding:20px;margin:18px 0}.big{font-size:30px;font-weight:700}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;background:white}td,th{padding:14px;border-bottom:1px solid #e3e8ee;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f7;padding:14px;font-size:13px}summary{cursor:pointer;font-weight:600}.note{color:#5a6c81}@media(max-width:700px){.cards,.pair{grid-template-columns:1fr}body{padding:16px}}
</style><h1>Qwen Pixel · Agent SFT / RL</h1>
<p>真实照片 + 编辑指令 → 模型调用 Python 读取和修改 RGB → 查看结果 → 提交 PNG 与像素数组。</p>
<p class="note">全部示例来自实际执行。有效提交与编辑质量分别评估；未完成实验不计为成功。当前是 COCO 照片的自定义小规模留出实验，不能当作官方 benchmark 分数或 RL 显著提升的证据。</p>
<div class="cards"><div class="card"><p>教师任务</p><div class="big">''' + str(teacher_count) + '''</div></div><div class="card"><p>教师有效提交</p><div class="big">''' + str(teacher_valid) + '''</div></div><div class="card"><p>教师判图通过</p><div class="big">''' + str(teacher_pass) + '''</div></div></div>
<h2>同一环境下的评测进度</h2><table><tr><th>模型阶段</th><th>已评测</th><th>有效提交</th><th>判图通过</th></tr>''' + table + '''</table>
<article><h3>RL 更新证据</h3><pre>''' + html.escape(json.dumps(rl, ensure_ascii=False, indent=2)) + '''</pre></article>
<h2>实际图片和工具操作</h2>''' + ''.join(examples) + '''<article><h3>何时扩大训练？</h3><p>先要求轨迹和图像数据无泄漏、SFT 与 RL 有真实权重更新，并在 20 张留出图上检查有效提交和编辑成功率。初步目标是有效提交 ≥90%、判图通过 ≥60%。这只支持扩大当前任务族；复杂物体替换和添加需要另行验证。</p></article></html>'''
    (OUT / 'index.html').write_text(page)
    (OUT / 'status.json').write_text(json.dumps({'teacher_tasks': teacher_count,
        'teacher_valid': teacher_valid, 'teacher_pass': teacher_pass,
        'evaluations': evaluations, 'rl': rl}, ensure_ascii=False, indent=2))
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
