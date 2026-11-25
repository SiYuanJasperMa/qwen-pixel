"""Current data, controlled SFT and same-task RL evidence for the live page."""
import html
import json
from pathlib import Path

from summarize_current_editing import collect, rows
from audit_paired_sft_exposure import collect_exposure


def render_current(root, out):
    evidence=collect()
    inventory=evidence['inventory'];rl=evidence['rl']
    tables=''.join(f'<tr><td>{html.escape(r["dataset"])}</td><td>{r["rft"]:,}</td><td>{r["ti"]:,}</td><td>{r["total"]:,}</td></tr>' for r in inventory['datasets'])
    section=f'<section class="card" id="current-editing"><h2>当前主线：训练 one-shot Pixel-Edit 模型</h2><p>模型输入原图和英文编辑指令，一次输出计划与 Python，程序从原图计算修改后的 RGB。下面统计训练样本行数，RFT + TI 可以直接相加。</p><div class="big">{inventory["total"]:,} 行英文 SFT 库存</div><p>RFT {inventory["rft"]:,} + TI {inventory["ti"]:,} · 独立任务 {inventory["unique_tasks"]:,}。库存包含不同 teacher 和增强版本，冻结训练快照需另做相同监督去重与质量审核。</p><div class="scroll"><table><thead><tr><th>来源</th><th>RFT</th><th>TI</th><th>合计</th></tr></thead><tbody>{tables}</tbody></table></div></section>'
    section+='<section class="card" id="rl-current"><h2>最新 RL smoke：同一批 64 个 MagicBrush dev 任务</h2>'
    if rl.get('matched_tasks'):
        section+=f'<div class="grid"><div><p>SFT 起点</p><div class="big">{rl["pre"]["pass"]}/64</div><p>代码执行 {rl["pre"]["executed"]}/64</p></div><div><p>RL 后</p><div class="big">{rl["post"]["pass"]}/64</div><p>代码执行 {rl["post"]["executed"]}/64</p></div></div><p>{rl.get("optimizer_updates",0)} 次真实参数更新，初始 SFT reference 权重不变。{rl["gained"]} 个任务由失败变成功，{rl["lost"]} 个由成功变失败。</p><p class="notice">这是自建图像 judge 的语义评估，不是官方像素指标。目前只有一个训练 seed，旧 greedy 推理存在数值重复性问题。正在用固定数值设置重复评测；不能直接归因为稳定 RL 收益。</p>'
    else:section+='<p>等待同一组完整前后结果。</p>'
    section+='</section><section class="card" id="matched-sft"><h2>正在推进：同 teacher、同任务的 RFT/TI 配比实验</h2><p>每个配置覆盖相同的 256 个真实训练任务，TI 替换 RFT 答案。0%、25%、50% 是样本行数比例，三组总回答 token 预算均为 100 万；实际 TI token 比例另行统计。</p>'
    experiment=root/'outputs/paired_method_sft_20261125'
    exposures=collect_exposure()if(experiment/'launch.json').exists()else{}
    progress=[]
    for name in ('ti00','ti25','ti50'):
        training=rows(experiment/name/'train/metrics.jsonl')
        last=training[-1]if training else {}
        summary=experiment/name/'eval/summary.json'
        evaluation=json.loads(summary.read_text())if summary.exists()else{}
        tokens=last.get('supervised_tokens',0)
        section+=f'<p><b>{name}</b> · {last.get("step",0):,} 步 · {tokens:,} / 1,000,000 回答 tokens · loss {last.get("loss","待启动")}</p><div class="bar"><i style="width:{min(100,100*tokens/1000000):.1f}%"></i></div>'
        if evaluation:section+=f'<p>固定 dev64：图像 judge {evaluation["judge_pass"]}/{evaluation["tasks"]}，代码执行 {evaluation["executed"]}/{evaluation["tasks"]}。</p>'
        exposure=exposures.get(name,{})
        if exposure.get('ti_token_fraction')is not None:
            section+=f'<p class="muted">实际TI回答token占比 {100*exposure["ti_token_fraction"]:.1f}% · 平均每行出现 {exposure["mean_visits_per_dataset_row"]:.2f} 次</p>'
        progress.append({'treatment':name,'step':last.get('step',0),'supervised_tokens':tokens,'eval':evaluation,'exposure':exposure})
    section+='<p class="muted">此实验限于 RFT 和 TI 都成功的任务，测试监督形式；TI 救回难任务的覆盖收益需要另一组对照。官方 benchmark、独立人工审核、跨数据集与多 seed 结果仍待补齐。</p></section>'
    section+='<section class="card"><h2>当前重复验证</h2><p>固定数值设置和6k输出预算，重新测同一批dev64；第二个RL seed采用相同设置。推理完成后展示全量结果。</p><div class="scroll"><table><thead><tr><th>流程</th><th>已评任务</th><th>执行成功</th><th>图像judge通过</th></tr></thead><tbody>'
    for key,label in [('rl_initial_deterministic','SFT起点'),('rl_updated_deterministic','已有RL模型'),('zero_shot_deterministic','Qwen3.8零样本')]:
        path=experiment/key
        completed=rows(path/'results.jsonl')
        if not completed:
            log=experiment/(key+'.log')
            observed={}
            if log.exists():
                for line in log.read_text().splitlines():
                    if not line.startswith('{'):continue
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue
                    if 'id'in r and 'exec'in r and 'pass'in r:
                        observed[r['id']]={'execution':{'status':r['exec']},'judge':{'pass':r['pass']}}
            completed=list(observed.values())
        if not completed and path.exists():
            # The runner writes final results at the end; do not infer passes
            # from the existence of an output image or a model-ready log line.
            done='推理中'
        else:done=str(len(completed))
        executed=sum((r.get('execution')or{}).get('status')=='passed'for r in completed)
        passed=sum((r.get('judge')or{}).get('pass')is True for r in completed)
        section+=f'<tr><td>{label}</td><td>{done}/64</td><td>{executed}</td><td>{passed}</td></tr>'
    section+='</tbody></table></div></section>'
    evidence['paired_sft']=progress
    (out/'current-editing.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    return section,evidence
