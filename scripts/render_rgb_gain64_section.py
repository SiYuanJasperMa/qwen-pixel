"""Live dashboard for the expanded paired validation study."""
import html
import json
from rgb_gain_summary import collect


def render_gain64(root,thumb=None):
    folder=root/'outputs/gpt61_rgb_gain_val64_v1'
    manifest=root/'data/gpt61_rgb_gain_val64_v1/manifest.json'
    if not manifest.is_file():return'',{}
    summary=collect(folder)
    panel='<section class="card" id="rgb-gain64"><h2>扩大验证：64题 image-only 与 image+RGB 配对对照</h2><p>MagicBrush 16 · UltraEdit 16 · SciForma 16 · MaskEdit中英文信息图各8。留出集共64个原图分组；每题两次独立 one-shot，总128次生成，结果不进入SFT。8并发，high reasoning，8192输出预算，两组均使用相同128×128源像素。</p><div class="scroll"><table><thead><tr><th>Setting</th><th>完成</th><th>代码执行</th><th>图像judge通过 / 已评估</th><th>API失败 / judge待评</th><th>平均输入 / 输出tokens</th></tr></thead><tbody>'
    for mode,label in [('image_only','只给image'),('image_rgb','image + RGB')]:
        s=summary['modes'][mode]
        panel+=f'<tr><td>{label}</td><td>{s["completed"]}/64</td><td>{s["executed"]}</td><td>{s["passed"]}/{s["evaluated"]}</td><td>{s["api_failed"]}/{s["judge_pending"]}</td><td>{s.get("mean_input_tokens",0):,.0f} / {s.get("mean_output_tokens",0):,.0f}</td></tr>'
    panel+='</tbody></table></div>'
    delta=summary['net_gain_pp']
    panel+=f'<p>完成有效配对 <b>{summary["paired_evaluated"]}/64</b>。只有RGB组成功：<b>{summary["rgb_only_wins"]}</b>题；只有image组成功：<b>{summary["image_only_wins"]}</b>题。'
    if delta is not None:panel+=f'净差 {delta:+.1f} 个百分点，配对精确检验 p={summary["mcnemar_exact_p"]:.3f}。'
    interval=summary.get('paired_bootstrap_95ci_pp')
    if interval:panel+=f'配对bootstrap 95%区间 [{interval[0]:+.1f}, {interval[1]:+.1f}] 个百分点。'
    status='已完成全部64对。'if summary['complete']else'进行中的差异会变化，等完整配对结果后判断增益。'
    panel+='</p><p class="notice">'+status+' API失败独立统计；代码能执行不等于编辑成功。成功率使用自建VLM judge，输入原图、编辑结果与指令，检查是否完整修改目标对象；不提供数据集目标图。两组teacher生成也均不看目标图。这轮分辨率是128，尚不代表原生高分辨率效果。</p>'
    panel+='<h3>各数据集：只统计两组都完成的配对</h3><div class="scroll"><table><thead><tr><th>数据集</th><th>有效配对</th><th>image成功</th><th>image+RGB成功</th><th>净差（百分点）</th></tr></thead><tbody>'
    for ds,s in summary['by_dataset'].items():
        panel+=f'<tr><td>{html.escape(ds)}</td><td>{s["paired_evaluated"]}</td><td>{s["image_only_passed"]}</td><td>{s["image_rgb_passed"]}</td><td>{s["net_gain_pp"]:+.1f}</td></tr>'
    panel+='</tbody></table></div><p class="muted">judge为GPT-5.5：要求指令≥4/5、保持≥4/5、质量≥3/5，且修改范围完整并≥4/5。输出token包含reasoning。代理未单独暴露图片token/图片费用，当前不推算金额，也不假设缓存折扣。</p><p><a href="rgb-gain64-summary.json">下载配对统计JSON</a></p>'
    if summary.get('resized_output_cases'):
        panel+='<p class="muted">有输出自行放大的案例：保留模型原始输出和代码，judge将原图等比例放大后比较；辅助像素指标将结果缩回源尺寸。不将图片尺寸检查错误当成语义失败。</p>'
    if thumb:
        tasks=[json.loads(l)for l in(root/'data/gpt61_rgb_gain_val64_v1/tasks.jsonl').read_text().splitlines()if l.strip()]
        panel+='<h3>全部验证题：原图 / image结果 / image+RGB结果 / 数据集目标</h3><p class="muted">按冻结任务顺序展示，失败同样保留。展开可看两组完整输出和judge理由；目标图仅供本地比较和展示。</p>'
        for task in tasks:
            tid=task['id'];records={}
            for mode in ['image_only','image_rgb']:
                p=folder/mode/tid/'record.json'
                if p.is_file():records[mode]=json.loads(p.read_text())
            if not records:continue
            labels=[];figures=[];details=[]
            src=thumb(task['image'],f'gain64_{tid}_source')
            figures.append(f'<figure><img loading="lazy" src="{src}"><figcaption>原图</figcaption></figure>')
            for mode,label in [('image_only','image'),('image_rgb','image+RGB')]:
                r=records.get(mode,{})
                judged=r.get('judge',{});passed=judged.get('pass')is True
                pending=r.get('judge_pending')or(r.get('execution',{}).get('status')=='passed'and not judged)
                state='成功'if passed else'judge待评'if pending else'失败'if r else'运行中'
                labels.append(f'{label}：{state}')
                p=folder/mode/tid/'edited.png'
                if p.is_file():
                    url=thumb(p,f'gain64_{tid}_{mode}')
                    figures.append(f'<figure><img loading="lazy" src="{url}"><figcaption>{label} · {state}</figcaption></figure>')
                else:figures.append(f'<figure><p class="muted">{html.escape(state)}</p><figcaption>{label}</figcaption></figure>')
                raw=folder/mode/tid/'completion.txt'
                reason=judged.get('reason')or r.get('error')or r.get('execution',{}).get('error')or state
                details.append(f'<details><summary>{label} 的judge理由与完整输出</summary><p>{html.escape(reason)}</p><pre>{html.escape(raw.read_text()if raw.is_file()else"尚无输出")}</pre></details>')
            target=thumb(task['target_image'],f'gain64_{tid}_target')
            figures.append(f'<figure><img loading="lazy" src="{target}"><figcaption>数据集目标图（teacher未见）</figcaption></figure>')
            panel+=f'<details class="card"><summary>{html.escape(task["dataset"])} · {html.escape(tid)} · {" / ".join(labels)}</summary><p>{html.escape(task["instruction"])}</p><div class="images gain64-images">{"".join(figures)}</div>{"".join(details)}</details>'
    panel+='</section>'
    return panel,summary
