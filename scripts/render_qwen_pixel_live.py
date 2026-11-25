"""Readable live HF dashboard for the current one-shot SFT / teacher / RL runs."""
import datetime as dt
import html
import json
from pathlib import Path

from PIL import Image
from render_rgb_text_section import render_rgb_section
from render_editing_growth_section import render_growth
from render_rgb_teacher_section import render_teacher_rgb
from render_rgb_gain64_section import render_gain64
from render_current_editing_section import render_current

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen_pixel_live'


def read(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.is_file() else {}


def rows(path):
    p = ROOT / path
    if not p.is_file(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def thumb(path, name):
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    if not p.is_file(): return ''
    dest = OUT / 'assets' / f'{name}.jpg'
    with Image.open(p) as im:
        im = im.convert('RGB'); im.thumbnail((512, 512)); im.save(dest, quality=86)
    return 'assets/' + dest.name


def render():
    (OUT / 'assets').mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间')
    sft = []
    for name, title, count in [('star_comparison_rft_v1', '纯 RFT', 8601),
                               ('star_comparison_mixed_v1', 'RFT + TI 混合', 12006)]:
        metrics = rows(f'outputs/{name}/train/metrics.jsonl')
        last = metrics[-1] if metrics else {}
        evaluation = read(f'outputs/{name}/eval/summary.json')
        curve = []
        for i in range(0, len(metrics), 10):
            window = metrics[max(0, i-29):i+1]
            curve.append([metrics[i]['supervised_tokens'], sum(r['loss'] for r in window)/len(window)])
        sft.append({'name': title, 'rows': count, 'step': last.get('step', 0),
                    'tokens': last.get('supervised_tokens', 0), 'eval': evaluation,
                    'curve': curve, 'complete': bool(evaluation)})
    teacher, sets = [], {}
    for folder, label in [('star_bulk_rft_gpt55_v1', 'GPT-5.5 · RFT'),
                           ('star_bulk_rft_gpt61_v1', 'GPT-6.1-Sol · RFT'),
                           ('star_bulk_ti_gpt61_v1', 'GPT-6.1-Sol · TI')]:
        records = rows(f'data/{folder}/judged.jsonl')
        records += rows(f'data/{folder.replace("star_bulk_", "star_bulk2_")}/judged.jsonl')
        target = 128 + (128 if (ROOT/'data/star_clean_v2/bulk_retry_tasks_2.jsonl').is_file() else 0)
        qualified = [r for r in records if (r.get('judge') or {}).get('pass')
                     and not r.get('hardcoded') and r.get('reuse_exec') == 'passed']
        sets[folder] = {r['task_id'] for r in qualified}
        teacher.append({'name': label, 'done': len(records), 'target': target,
                        'pass': sum(bool((r.get('judge') or {}).get('pass')) for r in records),
                        'qualified': len(qualified),
                        'error': sum(r.get('status') == 'teacher_error' for r in records)})
    r55 = sets['star_bulk_rft_gpt55_v1']; r61 = sets['star_bulk_rft_gpt61_v1']; ti = sets['star_bulk_ti_gpt61_v1']
    data = {'updated': now, 'sft': sft, 'teacher': teacher,
            'new_unique_tasks': len(r55 | r61 | ti), 'ti_only_same_teacher': len(ti-r61),
            'ti_only_all_teachers': len(ti-(r55|r61)),
            'rl': read('outputs/star_rl_smoke_lowtemp_v1/summary.json')}
    replicas = []
    for mode, title in [('rft', '纯 RFT'), ('mixed', 'RFT + TI')]:
        records = rows(f'outputs/star_comparison_{mode}_seed43/train/metrics.jsonl')
        last = records[-1] if records else {}
        replicas.append({'name': title, 'step': last.get('step', 0),
                         'tokens': last.get('supervised_tokens', 0)})
    data['replications'] = replicas
    replication_html = '<section class="card"><h2>现在正在跑：重复训练（seed 43）</h2><p class="muted">相同数据、相同 800 万回答 token 预算，检查第一轮多出的 3 个成功样本能否重复。</p>'
    for r in replicas:
        replication_html += f'<p><b>{r["name"]}</b> · {r["step"]:,} 步 · {r["tokens"]:,} / 8,000,000 tokens</p><div class="bar"><i style="width:{min(100,100*r["tokens"]/8000000):.1f}%"></i></div>'
    replication_html += '</section>'
    cards = []
    for r in sft:
        ev = r['eval']; n = ev.get('tasks', 0); passed = ev.get('judge_pass', 0)
        rate = f'{100*passed/n:.1f}%' if n else '评测中'
        cards.append(f'''<article class="card"><span class="pill">{'训练 + 评测完成' if r['complete'] else '进行中'}</span>
<h2>{r['name']}</h2><div class="big">{rate}</div><p>图像编辑成功：<b>{passed}/{n or 64}</b> 张</p>
<div class="bar"><i style="width:{100*min(1,r['tokens']/8000000):.1f}%"></i></div>
<p class="muted">训练 {r['tokens']:,} / 8,000,000 回答 tokens · {r['step']:,} 步<br>{r['rows']:,} 条训练轨迹 · 代码执行 {ev.get('executed',0)}/{n or 64}</p></article>''')
    teacher_rows = ''.join(f'''<tr><td>{r['name']}</td><td>{r['done']}/{r['target']}<div class="bar"><i style="width:{min(100,100*r['done']/r['target'])}%"></i></div></td><td>{r['pass']}</td><td>{r['qualified']}</td><td>{r['error']}</td></tr>''' for r in teacher)
    gallery = []
    tasks = {r['id']: r for r in rows('data/star_rl_smoke_lowtemp_v1/tasks.jsonl')}
    for j in (2, 3):
        base = f'outputs/star_rl_smoke_lowtemp_v1/group_00/sample_{j:02d}'
        outcome = read(base + '/outcome.json')
        if not outcome: continue
        task = tasks.get(outcome['task_id'], {})
        before = thumb(task.get('image', ''), f'rl{j}_source')
        after = thumb(outcome.get('execution', {}).get('output', ''), f'rl{j}_edited')
        raw = ROOT / base / 'completion.txt'
        try: code = json.loads(raw.read_text()).get('executable_code', '')
        except Exception: code = raw.read_text() if raw.exists() else ''
        gallery.append(f'''<article class="card"><h3>{html.escape(task.get('instruction',''))}</h3><p class="muted">SFT 模型在 RL 更新前独立采样 · 一次输出代码 · 图像 judge 通过</p><div class="images"><figure><img src="{before}"><figcaption>原图</figcaption></figure><figure><img src="{after}"><figcaption>模型代码编辑后</figcaption></figure></div><details><summary>查看模型输出的完整代码</summary><pre>{html.escape(code)}</pre></details></article>''')
    record = read('outputs/star_ti_example_v1/record.json')
    if record:
        src = thumb(record['source_image'], 'ti_source'); edited = thumb(ROOT/'outputs/star_ti_example_v1/edited.png', 'ti_edited'); target = thumb(record['target_image'], 'ti_target')
        gallery.append(f'''<article class="card"><h3>Teacher TI：移除摩托车后座的人</h3><p class="muted">GPT-6.1-Sol 看原图 + 目标图构造代码；执行时只用原图。该任务的 source-only RFT 对照失败。</p><div class="images triple"><figure><img src="{src}"><figcaption>原图</figcaption></figure><figure><img src="{edited}"><figcaption>执行 teacher 代码</figcaption></figure><figure><img src="{target}"><figcaption>数据集目标图</figcaption></figure></div><details><summary>查看 TI 编辑计划与代码</summary><pre>{html.escape(json.dumps(record.get('reasoning',{}),ensure_ascii=False,indent=2))}\n\n{html.escape(record.get('code',''))}</pre></details></article>''')
    rl = data['rl']; updates = rl.get('optimizer_updates',0)
    detail = next((r for r in rl.get('groups_detail',[]) if r.get('updated')), {})
    payload = json.dumps({'sft': sft, 'updated': now}, ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Pixel-Edit · 实验进度</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141d32;--text:#e9eef8;--muted:#9caec9;--line:#293651;--accent:#70e0c1}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:32px 22px}h1{font-size:40px;line-height:1.2;margin:10px 0}h2{font-size:23px}h3{font-size:18px}.muted{color:var(--muted)}.eyebrow{color:var(--accent);letter-spacing:2px;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px;margin:15px 0}.big{font-size:48px;color:var(--accent);font-weight:700;line-height:1.2}.pill{background:#193c38;color:#9bf0d6;border-radius:30px;padding:5px 12px;font-size:12px}.bar{height:7px;background:#29364e;border-radius:6px;overflow:hidden;margin-top:12px}.bar i{display:block;height:100%;background:var(--accent)}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:13px 10px;border-bottom:1px solid var(--line)}.scroll{overflow:auto}canvas{width:100%;height:240px;background:#10182b;border-radius:12px}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.triple{grid-template-columns:repeat(3,1fr)}figure{margin:0}img{width:100%;display:block;border-radius:10px;object-fit:contain;background:#080d17}figcaption{text-align:center;color:var(--muted);font-size:13px;padding:8px}pre{white-space:pre-wrap;word-break:break-word;max-height:480px;overflow:auto;background:#0b1222;padding:18px;border-radius:10px;font:12px/1.6 ui-monospace,monospace}summary{cursor:pointer;color:var(--accent)}.notice{border-left:3px solid var(--accent);padding-left:16px}.links a{color:var(--accent)}@media(max-width:700px){h1{font-size:30px}.grid{grid-template-columns:1fr}.card{padding:16px}.triple{grid-template-columns:1fr}.big{font-size:40px}}
</style></head><body><main><div class="eyebrow">QWEN PIXEL-EDIT · ONE-SHOT</div><h1>现在进展到哪里了？</h1><p>输入格式已统一，RL 已完成真实更新，新一轮 SFT 对照已出结果。</p><p class="muted">更新时间：__TIME__ · <button onclick="location.reload()">刷新</button> · 自动刷新 120 秒</p>
<div class="grid">__CARDS__</div><p class="notice">这轮混合数据多成功 3 张，是初步信号。两组只各训练了一次；数据组成也不同，还不能认定提升稳定，或把差异完全归因于 TI。</p><p class="muted">评测口径：同一批 64 张官方 MagicBrush dev 图片，使用自建 VLM judge 检查指令、区域保持和视觉质量。这里的百分比不是官方 benchmark 的像素指标。</p>
<section class="card"><h2>训练 loss 曲线</h2><p class="muted">横轴：累计回答 tokens（百万） · 纵轴：最近 30 步平均 loss。两组训练样本不同，loss 仅用于观察训练是否正常。</p><canvas id="curve"></canvas><p><span style="color:#70e0c1">● 纯 RFT</span>　<span style="color:#a5a0ff">● RFT + TI</span></p></section>
<section class="card"><h2>Teacher 数据构造：当前批次</h2><p>同一批 128 个曾失败的任务，比较 source-only RFT 与 target-informed TI；完成 __DONE__ / 384 次生成。</p><div class="scroll"><table><thead><tr><th>模型 / 构造方式</th><th>本批进度</th><th>图像 judge 通过</th><th>合格轨迹</th><th>API 错误</th></tr></thead><tbody>__TEACHERS__</tbody></table></div><p>合并后：<b>__UNIQUE__ 个合格独立任务</b>。同一 GPT-6.1 对照中，TI 额外救回 <b>__TI_ONLY__ 个</b>；相对全部 RFT teacher 的额外补救为 <b>__TI_ALL__ 个</b>。</p><p class="muted">“合格”还要求程序过滤和执行检查通过。合并时按任务去重，不把多 teacher 的多条轨迹当成多个新任务。这里是本批进度，后续数据仍需继续构造。</p></section>
<section class="card"><h2>RL smoke：已跑通训练环</h2><p>真实任务采样 → 执行 Python → 看实际结果图打分 → 计算组内优势 → 更新 LoRA。</p><div class="big">__UPDATES__ 次真实更新</div><p>低温度测试的奖励是 <b>[0, 0, 1, 1]</b>；4 次独立编辑中有 2 次图像 judge 通过。梯度范数 __GRAD__，权重变化已核实。</p><p class="notice">这些编辑是在更新前采样出来的。它们证明模型有可用起点和 RL 奖励信号；不代表 RL 后 benchmark 达到 50%，也不证明一次更新提升了性能。</p></section>
<h2>直接看编辑结果与代码</h2>__GALLERY__<section class="card"><h2>下一步</h2><p>重复训练确认这轮混合数据的收益；扩大合格数据与失败任务补救；用修正后的 SFT 模型开展更长的 RL，并在独立留出集上做更新前后对照。</p><p class="muted">已修复训练 / 推理模板及多模态 token 回放，按原图与重复 RGB 内容分组。10 步 SFT 工程测试通过权重更新检查，2 条留出样本的执行仍失败；不将工程测试当成模型质量结果。</p></section><p class="links"><a href="https://huggingfriends-magicbrush-teacher-progress.static.hf.space/index.html">查看旧实验看板</a> · <a href="progress.json">下载当前进度 JSON</a></p></main>
<script>const state=__PAYLOAD__;const c=document.getElementById('curve');function draw(){let d=devicePixelRatio||1,w=c.clientWidth,h=240;c.width=w*d;c.height=h*d;let x=c.getContext('2d');x.scale(d,d);let left=44,top=14,right=w-15,bottom=h-32;x.strokeStyle='#33425e';x.fillStyle='#9caec9';x.font='12px system-ui';for(let i=0;i<=4;i++){let y=top+(bottom-top)*i/4;x.beginPath();x.moveTo(left,y);x.lineTo(right,y);x.stroke();x.fillText((.8-i*.15).toFixed(2),3,y+4)}for(let i=0;i<=4;i++)x.fillText(String(i*2),left+(right-left)*i/4,bottom+22);state.sft.forEach((r,i)=>{x.strokeStyle=i?'#a5a0ff':'#70e0c1';x.lineWidth=2;x.beginPath();r.curve.forEach((p,j)=>{let px=left+(right-left)*Math.min(1,p[0]/8000000),py=bottom-(bottom-top)*(p[1]-.2)/.6;j?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()})}draw();addEventListener('resize',draw);setTimeout(()=>location.reload(),120000);</script></body></html>'''
    replacements = {'__TIME__': now, '__CARDS__': ''.join(cards), '__TEACHERS__': teacher_rows,
        '__DONE__': str(sum(r['done'] for r in teacher)), '__UNIQUE__': str(data['new_unique_tasks']),
        '__TI_ONLY__': str(data['ti_only_same_teacher']), '__TI_ALL__': str(data['ti_only_all_teachers']),
        '__UPDATES__': str(updates), '__GRAD__': f"{detail.get('grad_norm',0):.4f}",
        '__GALLERY__': ''.join(gallery), '__PAYLOAD__': payload}
    page = page.replace('<section class="card"><h2>训练 loss 曲线</h2>', replication_html + '<section class="card"><h2>训练 loss 曲线</h2>')
    page = page.replace(' / 384 次生成。', f' / {sum(r["target"] for r in teacher)} 次生成。')
    for key, value in replacements.items(): page = page.replace(key, value)
    rgb_html, rgb_status = render_rgb_section(ROOT, OUT, thumb)
    page = page.replace('<div class="grid">', rgb_html + '<div class="grid">', 1)
    page = page.replace('</style>', '.rgb-images{grid-template-columns:repeat(5,1fr)}a{color:var(--accent)}@media(max-width:700px){.rgb-images{grid-template-columns:1fr 1fr}} </style>')
    data['rgb_text'] = rgb_status
    growth_html,growth_status=render_growth(ROOT,OUT,thumb)
    page=page.replace('<section id="rgb-text"',growth_html+'<section id="rgb-text"',1)
    page=page.replace('Teacher 数据构造：当前批次','Teacher：之前两批的固定对照结果')
    data['editing_growth']=growth_status
    teacher_rgb_html,teacher_rgb_status=render_teacher_rgb(ROOT,thumb)
    page=page.replace('<section id="rgb-text"',teacher_rgb_html+'<section id="rgb-text"',1)
    data['gpt61_rgb_smoke']=teacher_rgb_status
    gain64_html,gain64=render_gain64(ROOT,thumb)
    page=page.replace('<section class="card" id="gpt61-three-settings">',gain64_html+'<section class="card" id="gpt61-three-settings">',1)
    data['gpt61_rgb_gain64']=gain64
    page=page.replace('</style>','.gain64-images{grid-template-columns:repeat(4,1fr)}@media(max-width:700px){.gain64-images{grid-template-columns:1fr 1fr}}</style>')
    (OUT/'rgb-gain64-summary.json').write_text(json.dumps(gain64,ensure_ascii=False,indent=2))
    style=page.split('<style>',1)[1].split('</style>',1)[0]
    focused='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-6.1 RGB增益配对验证</title><style>'+style+'</style></head><body><main><h1>GPT-6.1：额外输入RGB有没有增益？</h1><p class="muted">'+now+' · <a href="index.html">返回SFT与teacher进度</a> · <button onclick="location.reload()">刷新</button></p>'+gain64_html+'</main><script>setTimeout(()=>location.reload(),120000)</script></body></html>'
    (OUT/'rgb-gain64.html').write_text(focused)
    current_html,current_status=render_current(ROOT,OUT)
    page=page.replace('<div class="grid">',current_html+'<div class="grid">',1)
    page=page.replace('<h2>RL smoke：已跑通训练环</h2>','<h2>早期 RL 工程检查（历史结果）</h2>')
    data['current_editing']=current_status
    current_page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Pixel-Edit 当前训练进度</title><style>'+style+'</style></head><body><main><h1>Qwen Pixel-Edit：当前训练进度</h1><p class="muted">'+now+' · <a href="index.html">完整实验看板</a></p>'+current_html+growth_html+'</main><script>setTimeout(()=>location.reload(),120000)</script></body></html>'
    (OUT/'editing-progress.html').write_text(current_page)
    (OUT / 'index.html').write_text(page)
    # Only public metrics, not full model configurations or credentials.
    data['rl'] = {k: rl.get(k) for k in ('groups','optimizer_updates','nonzero_update_verified')}
    (OUT / 'progress.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({'updated': now, 'teacher': teacher, 'unique': data['new_unique_tasks'],
                      'ti_only': data['ti_only_same_teacher']}, ensure_ascii=False))


if __name__ == '__main__': render()
