"""Show the GPT-6.1 text-only RGB smoke and its independent control."""
import html
import json
from pathlib import Path


def render_teacher_rgb(root,thumb):
    folder=root/'outputs/rgb_text_gpt61_smoke_v1'
    requests=list(folder.glob('*/*/request.json'))
    if not requests:return'',{}
    records=[]
    for path in folder.glob('*/*/record.json'):
        try:records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:pass
    modes={}
    for mode in ['rgb_text','instruction_only']:
        subset=[r for r in records if r['mode']==mode]
        modes[mode]={'completed':len(subset),'expected':3,
                     'executed':sum(r.get('execution',{}).get('status')=='passed'for r in subset),
                     'judge_pass':sum(r.get('judge',{}).get('pass')is True for r in subset),
                     'judge_pending':sum(bool(r.get('judge_pending'))for r in subset)}
    text='<section class="card" id="gpt61-rgb"><h2>GPT‑6.1‑Sol：只读 RGB 文本的真实编辑 smoke</h2><p>模型输入只有编辑指令和完整 RGB 数值文本，没有图片、目标图或工具；一次生成 Python，随后执行并判分。源图与代码执行均为 128×128，与 Qwen smoke 是同一批任务。</p><div class="scroll"><table><thead><tr><th>条件</th><th>完成</th><th>代码执行</th><th>图像 judge 通过</th></tr></thead><tbody>'
    for mode,label in [('rgb_text','RGB 文本 + 指令'),('instruction_only','仅指令对照')]:
        s=modes[mode];text+=f'<tr><td>{label}</td><td>{s["completed"]}/3</td><td>{s["executed"]}</td><td>{s["judge_pass"]}</td></tr>'
    text+='</tbody></table></div><p class="notice">这是三张真实图片的 smoke，使用自建图像 judge。仅指令对照同样可以生成在运行时读取像素的代码；两组对比用于检查提供 RGB 文本是否带来额外帮助。GPT‑6.1 使用 high reasoning、8192 输出预算，和 Qwen 的计算预算不同。所有样本来自留出集，结果不会加入训练数据。</p></section>'
    for record in sorted(records,key=lambda r:(r['id'],r['mode'])):
        if record['mode']!='rgb_text':continue
        output=record.get('execution',{}).get('output')
        if not output or not Path(output).is_file():continue
        source=thumb(record['source_image'],f'g61rgb_{record["id"]}_source')
        edited=thumb(output,f'g61rgb_{record["id"]}_edited')
        target=thumb(record['target_image'],f'g61rgb_{record["id"]}_target')
        usage=record.get('teacher_usage')or{}
        completion=folder/'rgb_text'/record['id']/'completion.txt'
        code=completion.read_text()if completion.is_file()else''
        reason=(record.get('judge')or{}).get('reason','')
        text+=f'<article class="card"><h3>{html.escape(record["instruction"])}</h3><p class="muted">实际输入 {usage.get("input_tokens",0):,} tokens · API 输出 {usage.get("output_tokens",0):,} tokens（含 reasoning，见 usage）· 无图片输入</p><div class="images triple"><figure><img src="{source}"><figcaption>原图，模型只收到其 RGB 文本</figcaption></figure><figure><img src="{edited}"><figcaption>GPT‑6.1 单次代码执行结果</figcaption></figure><figure><img src="{target}"><figcaption>目标图，仅用于本地比较</figcaption></figure></div><p>{html.escape(reason)}</p><details><summary>完整可见回答与代码</summary><pre>{html.escape(code)}</pre></details></article>'
    comparison=root/'outputs/gpt61_three_settings_v1/comparison.json'
    if comparison.is_file():
        report=json.loads(comparison.read_text())
        panel='<section class="card" id="gpt61-three-settings"><h2>GPT‑6.1‑Sol：三个输入 setting 的同任务对照</h2><p>同三张源图、128×128、high reasoning、一次输出代码；三个条件分别是 image、image+RGB 文本、RGB 文本。目标图都不进入模型输入。</p><div class="scroll"><table><thead><tr><th>Setting</th><th>图像 judge</th><th>平均输入 tokens</th><th>平均输出 tokens（含 reasoning）</th><th>平均可见输出 tokens</th></tr></thead><tbody>'
        for mode,label in [('image_only','只给 image'),('image_rgb','image + RGB'),('rgb_text','只给 RGB')]:
            s=report['settings'][mode]
            panel+=f'<tr><td>{label}</td><td>{s["judge_pass"]}/{s["tasks"]}</td><td>{s["mean_input_tokens"]:,.0f}</td><td>{s["mean_output_tokens"]:,.0f}</td><td>{s["mean_visible_output_tokens"]:,.0f}</td></tr>'
        panel+='</tbody></table></div><p class="muted">按用户最新要求不考虑缓存。输入数是代理上报的 usage；接口没有单列图片 token/图片费用，且尚未取得此账号的实际计费单价，因此这里不编造美元价格。费用公式为 input_tokens × 输入单价 + output_tokens × 输出单价（单价按每 token 换算）。这是三例 smoke，不是完整 benchmark。</p></section>'
        # Show source and all three edited results for direct visual comparison.
        source_tasks=[json.loads(l)for l in(root/'data/rgb_text_smoke_v1/tasks.jsonl').read_text().splitlines()]
        for task in source_tasks:
            figures=[]
            image=thumb(task['image'],f'three_{task["id"]}_source')
            figures.append(f'<figure><img src="{image}"><figcaption>原图</figcaption></figure>')
            for mode,label in [('image_only','只给 image'),('image_rgb','image + RGB'),('rgb_text','只给 RGB')]:
                base=root/'outputs/gpt61_three_settings_v1'/mode/task['id']
                image=thumb(base/'edited.png',f'three_{task["id"]}_{mode}')
                figures.append(f'<figure><img src="{image}"><figcaption>{label}</figcaption></figure>')
            panel+=f'<article class="card"><h3>{html.escape(task["instruction"])}</h3><div class="images" style="grid-template-columns:repeat(4,1fr)">{"".join(figures)}</div></article>'
        text=panel+text
    return text,{'model':'traex/GPT-6.1-Sol','image_inputs':0,'target_in_model_input':False,'modes':modes,
                 'three_settings':report if comparison.is_file()else None}
