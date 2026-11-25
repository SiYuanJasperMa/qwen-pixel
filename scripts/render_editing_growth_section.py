"""Show live, continuing data production independently from frozen experiments."""
import html
import json
import hashlib
from collections import Counter,defaultdict
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pixel.data_language import english_task,english_supervision
from pixel.data_quality import review_registry,task_excluded,strong_image_grade
from pixel.source_policy import source_disabled,disabled_sources


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return{}


def render_growth(root,out,thumb):
    folder=root/'data/editing_growth_v1'
    state=read(folder/'status.json')
    if not state:return'',{}
    ingest=read(folder/'ingestion_status.json'); counts=read(folder/'ingestion_state.json').get('counts',{})
    snapshot=read(folder/'sft/manifest.json')
    aliases={'gpt55':'GPT-5.5','gpt61':'GPT-6.1-Sol','gpt62':'GPT-6.2-Sol','terra':'GPT-5.6-Terra','luna':'GPT-5.6-Luna'}
    table=''
    for lane in state.get('lanes',[]):
        table+=f'<tr><td>{aliases.get(lane["teacher"],lane["teacher"])} · {lane["mode"].upper()}</td><td>{lane["running_or_waiting_api"]} / {lane["concurrency"]}</td><td>{lane["pending"]:,}</td><td>{lane["completed"]:,}</td><td>{lane["qualified"]:,}</td><td>{lane["retryable_errors"]}</td></tr>'
    phases={'indexing_existing_rgb':'核对原有训练 / 验证图片内容','downloading_training_shard':'读取新的官方训练分片',
            'reservoir_full':'任务池已足够，随生成进度继续补充','image_storage_budget':'图片存储预算已满，等待释放空间'}
    phase=phases.get(ingest.get('phase'),ingest.get('phase','准备接入'))
    section=f'<section class="card" id="data-growth"><h2>持续构造 SFT 数据</h2><p>任务池 <b>{state["task_pool"]:,}</b> 个 · 本次合格轨迹 <b>{state["accepted_trajectories"]:,}</b> 条 · 对应 <b>{state["accepted_unique_tasks"]:,}</b> 个任务</p><p class="muted">生产心跳：{html.escape(state["updated"])} · {"北京时间夜间高并发" if state.get("night_parallelism")else"白天并发配置"}。任务池会自动读取新数据，不在 128 条或 10k 行时停止。</p><div class="scroll"><table><thead><tr><th>Teacher / 方式</th><th>运行或等待 API / 并发上限</th><th>待处理</th><th>已完成</th><th>合格轨迹</th><th>待重试</th></tr></thead><tbody>{table}</tbody></table></div><p>自动扩库：<b>{phase}</b>。已接入任务：{html.escape(json.dumps(counts,ensure_ascii=False))}。</p><p>已导出 <b>{snapshot.get("rows",0):,}</b> 条新 SFT 轨迹。原有 8,601 / 12,006 条对照训练集保持冻结，新增数据单独保存，用于后续规模实验。</p><p class="notice">合格要求：一次 teacher 输出、程序过滤、执行、完整目标区域的图像 judge 和重用执行检查通过。API 或 judge 错误会重试，不计为成功样本。任务按原图内容分组，排除官方 MagicBrush dev 与现有验证图片的内容重叠。TI 在 source-only 失败后补救，学生输入只包含原图和指令。</p></section>'
    section='<section class="card"><h2>InstructPix2Pix 已停用</h2><p>按用户要求，不再采集或构造该来源的RFT/TI。已经生成的原始记录、图片与旧实验保留；历史库存数字包含这些存档，新训练快照排除该来源。</p></section>'+section
    independent=read(folder/'ti_balanced_en_v1/status.json')
    ti_export=read(folder/'ti_balanced_en_v1/sft/manifest.json')
    section+='<section class="card" id="ti-enrichment"><h2>英文 TI：独立补充训练样本</h2><p>RFT成功的任务也构造TI，保留两种方法的合格轨迹。新数据只保留英文任务和监督文本；旧实验保留用于复现，中文子集不进入新训练快照。北京时间工作日18:00–09:00、周末全天提高Sol并发。1:1为训练配比对照目标，当前通过率不被人为调整。</p>'
    if independent:
        section+=f'<p>独立TI合格样本 <b>{ti_export.get("rows",0):,}</b> · 生产心跳 {html.escape(independent["updated"])}</p><div class="scroll"><table><thead><tr><th>TI teacher</th><th>运行/并发上限</th><th>待处理</th><th>已完成</th><th>合格</th><th>待重试</th></tr></thead><tbody>'
        for lane in independent.get('lanes',[]):
            section+=f'<tr><td>{aliases.get(lane["teacher"],lane["teacher"])}</td><td>{lane["running_or_waiting_api"]}/{lane["concurrency"]}</td><td>{lane["pending"]:,}</td><td>{lane["completed"]:,}</td><td>{lane["qualified"]:,}</td><td>{lane["retryable_errors"]}</td></tr>'
        section+='</tbody></table></div>'
    else:section+='<p>TI流水线配置已准备，尚无运行心跳。</p>'
    counter=Counter();ids=set()
    night_state=read(folder/'ti_night_sol_v2/status.json')
    if night_state:
        section+=f'<h3>非工作时间补充：61/62-Sol TI</h3><p>状态 {html.escape(night_state["state"])} · 心跳 {html.escape(night_state["updated"])}。仅北京时间工作日18:00–09:00或周末发起新请求；白天保留排队中的调用。优先科研图、文字布局、区域编辑和MagicBrush。</p><div class="scroll"><table><thead><tr><th>Teacher</th><th>运行或等待 / 上限</th><th>待构造</th><th>合格轨迹</th></tr></thead><tbody>'
        for lane in night_state.get('lanes',[]):
            section+=f'<tr><td>{aliases.get(lane["teacher"],lane["teacher"])}</td><td>{lane["running_or_waiting_api"]}/{lane["concurrency"]}</td><td>{lane["pending"]:,}</td><td>{lane["qualified"]:,}</td></tr>'
        section+='</tbody></table></div><p class="muted">每个teacher增加至多16个夜间请求槽；额外候选使用独立ID，不改写旧轨迹。并发上限包含可能仍在API队列中的请求，不代表完成速度。</p>'
    for path in [root/'data/star_clean_v2/sft_mixed_aug/train.jsonl',folder/'sft/train.jsonl',folder/'gpt62/sft/train.jsonl',folder/'ti_balanced_en_v1/sft/train.jsonl',folder/'ti_night_sol_v2/sft/train.jsonl']:
        if not path.is_file():continue
        for line in path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if not english_task(row) or not english_supervision(row):continue
            meta=row['meta'];tid=meta['task_id'];ds=meta.get('dataset')or next((d for d in ('magicbrush','ultraedit','sciforma')if tid.startswith(d)),'unknown')
            counter[(ds,meta['generation_mode'])]+=1;ids.add(tid)
    section+='<h3>当前英文SFT库存：按样本行数统计，可相加</h3><div class="scroll"><table><thead><tr><th>数据来源</th><th>RFT行数</th><th>TI行数</th><th>合计</th></tr></thead><tbody>'
    for ds in sorted({ds for ds,mode in counter}):
        a,b=counter[(ds,'rft')],counter[(ds,'ti')]
        section+=f'<tr><td>{html.escape(ds)}</td><td>{a:,}</td><td>{b:,}</td><td>{a+b:,}</td></tr>'
    rft=sum(n for(ds,m),n in counter.items()if m=='rft');ti=sum(n for(ds,m),n in counter.items()if m=='ti')
    section+=f'</tbody></table></div><p>英文SFT行数 <b>{rft+ti:,}</b> = RFT {rft:,} + TI {ti:,}；独立任务覆盖 {len(ids):,}。行数包含不同teacher轨迹和增强版本，后续冻结训练快照另做完全相同监督内容的去重。</p></section>'
    records={}
    path=folder/'judged.jsonl'
    if path.is_file():
        for line in path.read_text().splitlines():
            try:r=json.loads(line);records[r['id']]=r
            except json.JSONDecodeError:pass
    review=review_registry()
    accepted=[r for r in records.values()if r.get('pipeline_qualified')and r.get('retained_output')and english_task(r)and english_supervision(r)
              and not source_disabled(r)and not task_excluded(r,review)and strong_image_grade(r.get('judge')or{})]
    section+='<section class="card"><h3>展示样本的质量门槛</h3><p>下方展示自建图像judge通过、画质与保留评分均至少4/5、且未被质量复核隔离的候选。上方库存保留原始生产判定；下一份训练快照单独过滤待审样本和低质量InstructPix2Pix。</p><p>InstructPix2Pix来自自动生成的编辑指令和图像对，包含生硬指令与目标漂移，需逐条审核后作为扩展数据。</p></section>'
    for i,r in enumerate(accepted[-3:]):
        key=hashlib.sha256(r['id'].encode()).hexdigest()[:12]
        source=thumb(r['source_image'],f'growth_{key}_source')
        edited=thumb(r['retained_output'],f'growth_{key}_edited')
        target=thumb(r['target_image'],f'growth_{key}_target')
        section+=f'<article class="card"><h3>质量筛选后的候选：{html.escape(r["instruction"])}</h3><p class="muted">{r["dataset"]} · {aliases.get(r["teacher"],r["teacher"])} · {r["generation_mode"].upper()} · 自建图像judge通过 · 画质{r["judge"]["visual_quality"]}/5 · 保留{r["judge"]["preservation"]}/5</p><div class="images triple"><figure><img src="{source}"><figcaption>原图</figcaption></figure><figure><img src="{edited}"><figcaption>单次 teacher 代码执行</figcaption></figure><figure><img src="{target}"><figcaption>数据集目标</figcaption></figure></div><details><summary>完整 teacher 计划与代码</summary><pre>{html.escape(json.dumps(r.get("reasoning",{}),ensure_ascii=False,indent=2))}\n\n{html.escape(r["code"])}</pre></details></article>'
    return section,{'production':state,'ingestion':ingest,'new_dataset_tasks':counts,'sft':snapshot,
        'independent_ti':independent,'independent_ti_sft':ti_export,'night_sol_ti':night_state,
        'english_sft_rows':{'rft':rft,'ti':ti,'total':rft+ti},'english_unique_tasks':len(ids)}
