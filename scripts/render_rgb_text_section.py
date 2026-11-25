"""Public, live status for the vision-free RGB input experiment."""
import html
import json
import shutil
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A producer can be writing its last line.
    return result


def render_rgb_section(root, out, thumb):
    tasks = read_rows(root / 'data/rgb_text_smoke_v1/tasks.jsonl')
    if not tasks:
        return '', {}
    labels = {'rgb_text': '纯 LLM + RGB 文本', 'instruction_only': '纯 LLM + 仅指令',
              'vl': 'VL + 图像'}
    runs = {'rgb_text': root / 'outputs/rgb_text_smoke_v1',
            'instruction_only': root / 'outputs/rgb_text_smoke_v1',
            'vl': root / 'outputs/rgb_text_vl_control_v1'}
    stats, records, table = {}, {}, []
    final_report = read_json(root / 'outputs/rgb_text_final_report.json')
    for mode, folder in runs.items():
        subset = [r for r in read_rows(folder / 'results.jsonl') if r['mode'] == mode]
        # Use the latest attempt if a run was explicitly restarted.
        records[mode] = {r['id']: r for r in subset}
        for r in read_rows(folder / 'evaluated_results.jsonl'):
            prior = records[mode].get(r['id'], {})
            if r['mode'] == mode and prior.get('source_rgb_sha256') == r.get('source_rgb_sha256'):
                records[mode][r['id']] = r
        for run in final_report.get('runs', {}).values():
            for r in run.get('records', []):
                if r['mode'] == mode and r['id'] in {t['id'] for t in tasks}:
                    records[mode][r['id']] = r
        subset = list(records[mode].values())
        audit = read_json(folder / 'model_audit.json')
        stats[mode] = {'completed': len(subset), 'expected': len(tasks),
                      'executed': sum(r.get('execution', {}).get('status') == 'passed' for r in subset),
                      'judge_pass': sum(bool(r.get('judge', {}).get('pass')) for r in subset),
                      'judge_pending': sum(bool(r.get('judge_pending')) or
                                           (r.get('execution', {}).get('status') == 'passed' and not r.get('judge'))
                                           for r in subset),
                      'errors': [r.get('error') for r in subset if r.get('error')],
                      'input_tokens': [r.get('input_tokens') for r in subset], 'model_audit': audit}
        s = stats[mode]
        state = '已完成' if s['completed'] == len(tasks) else ('已加载，测试中' if audit else '尚未记录加载结果')
        if s['errors']:
            state += '；有失败记录'
        tokens = ', '.join(f'{n:,}' for n in s['input_tokens'] if n is not None) or '待记录'
        table.append(f'<tr><td>{labels[mode]}</td><td>{state}<br>{s["completed"]}/{len(tasks)}</td>'
                     f'<td>{s["executed"]}</td><td>{s["judge_pass"]}</td><td>{s["judge_pending"]}</td><td>{tokens}</td></tr>')
    budget = read_json(root / 'outputs/text_rgb_budget/measurement.json')
    budgets = [b for b in budget.get('budgets', []) if b['encoding'] == 'hex_rle']
    budget_text = ' · '.join(f'{b["side"]}×{b["side"]}：{b["tokens"]:,} tokens' for b in budgets)
    galleries = []
    for task in tasks:
        tid = task['id']
        source = thumb(task['image'], f'rgb_{tid}_source')
        target = thumb(task['target_image'], f'rgb_{tid}_target')
        figures = [f'<figure><img src="{source}"><figcaption>原图 · {task["width"]}×{task["height"]}</figcaption></figure>']
        details = []
        for mode, folder in runs.items():
            r = records[mode].get(tid, {})
            edited = r.get('execution', {}).get('output')
            src = thumb(edited, f'rgb_{tid}_{mode}') if edited else ''
            label = labels[mode]
            if src:
                judge = 'judge 待评' if r.get('judge_pending') or not r.get('judge') else ('judge 通过' if r.get('judge', {}).get('pass') else 'judge 未通过')
                figures.append(f'<figure><img src="{src}"><figcaption>{label} · {judge}</figcaption></figure>')
            else:
                state = html.escape(r.get('error') or r.get('execution', {}).get('error') or '尚无输出图')
                figures.append(f'<figure><div class="muted" style="padding:25px 8px;background:#0b1222;border-radius:10px">{state}</div><figcaption>{label}</figcaption></figure>')
            raw_path = folder / mode / tid / 'completion.txt'
            if raw_path.is_file():
                details.append(f'<details><summary>{label}：完整单次输出（{r.get("output_tokens", "?")} tokens）</summary><pre>{html.escape(raw_path.read_text())}</pre></details>')
            if r.get('judge'):
                reason = html.escape(r['judge'].get('reason', ''))
                gain = r.get('metrics', {}).get('gain', 0)
                details.append(f'<details><summary>{label}：图像判分依据</summary><p>{reason}</p><p class="muted">输入 {r.get("input_tokens",0):,} tokens · 输出 {r.get("output_tokens",0):,} tokens · 运行 {r.get("elapsed_seconds",0):.1f} 秒 · 相对 no-op 的 L1 改善 {gain:.6f}</p></details>')
        figures.append(f'<figure><img src="{target}"><figcaption>目标图 · 仅用于评测</figcaption></figure>')
        rgb_path = Path(task['rgb_text'])
        dest = out / 'assets' / f'{tid}.rgb.txt'
        shutil.copyfile(rgb_path, dest)
        preview = rgb_path.read_text()[:1600]
        galleries.append(f'<article class="card"><h3>{html.escape(task["instruction"])}</h3>'
                         f'<p class="muted">{task["dataset"]} · RGB 输入本身 {task["rgb_tokens"]:,} tokens · 以下图片均由相同分辨率的源图产生</p>'
                         f'<div class="images rgb-images">{"".join(figures)}</div>'
                         f'<details><summary>查看真实 RGB 文本格式</summary><pre>{html.escape(preview)}\n…</pre>'
                         f'<a href="assets/{dest.name}" target="_blank">打开完整 RGB 输入</a></details>{"".join(details)}</article>')
    result = '<section id="rgb-text" class="card"><h2>新增实验：纯 LLM 读取 RGB 文本</h2>'
    result += '<p>输入：编辑指令 + 完整 RGB 数值文本 → 无视觉编码器的 Qwen 语言模型 → 一次输出 Python → 执行并渲染图像。</p>'
    result += '<p class="muted">本次模型为未挂 SFT adapter 的 Qwen3.8-27B。RGB 使用十六进制和逐行重复像素计数，可逐像素还原；不量化，不使用图片 tokenizer。语言权重来自同一个 checkpoint，加载时检查缺失权重和视觉模块。三个条件共用编辑要求和输出格式。</p>'
    result += f'<div class="scroll"><table><thead><tr><th>条件</th><th>进度</th><th>代码执行</th><th>图像 judge 通过</th><th>judge 待评</th><th>实际输入 tokens</th></tr></thead><tbody>{"".join(table)}</tbody></table></div>'
    result += f'<p>真实 MagicBrush 照片的 lossless RGB 文本长度：{budget_text}。</p>'
    result += '<p class="notice">本次为 3 个任务的工程与能力 smoke，全部缩放到 128×128；科研图的 512×512 输入超预算后明确降到 128×128。此结果不代表原生分辨率能力或 benchmark 成功率。模型输入不含目标图，三组执行使用同一源图。</p></section>'
    result += ''.join(galleries)
    return result, {'tasks': len(tasks), 'modes': stats, 'rgb_budget': budgets,
                    'resolution': '128x128 resized', 'target_in_model_input': False}
