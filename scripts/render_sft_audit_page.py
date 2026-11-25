"""Render measured SFT status and actual development image pairs locally."""
import datetime
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    return json.loads(path.read_text())if path.exists()else default


def main():
    out=ROOT/'outputs/sft_audit';out.mkdir(exist_ok=True)
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Pixel SFT</title>
<style>body{font-family:system-ui;max-width:1080px;margin:auto;padding:24px;background:#f4f7fb;color:#182839}article{background:white;padding:20px;margin:18px 0;border-radius:12px}p{line-height:1.65}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;max-height:520px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<h1>Qwen Pixel · SFT</h1><p>只做真实照片的 Python / RGB 编辑。Goal 仍在进行；尚未达到可靠未见图像编辑。RL 暂不开展。</p>
<article><h2>已验证与待验证</h2><p>训练 / 推理序列化、assistant-only 移位 loss、历史图像快照、LoRA 更新和 checkpoint processor 张量一致性已核对。单轨迹 4/4、三轨迹 10/10 完整调用及三张训练图 RGB 精确复现通过。</p><p>未见图片结果：旧 ROI 0/3；clean v2 checkpoint400 严格通过 1/8，工具完成 8/8；同接口 Instruct 基线严格通过 0/8，工具完成 5/8。低 loss 与成功写文件不足以证明会编辑。</p><p>64 个训练 mask 中 40 个通过完整目标 / 排除无关物体审核。v3 重做 gamma0.75 亮度、25% tint 的实际程序和工具反馈，再筛选全部参考编辑。八张开发图和二十张最终 COCO val2017 图固定隔离。</p></article>''']
    review=read(ROOT/'data/sft_clean_mask_v3/reference_review_manifest.json')
    if review:
        parts.append('<article><h2>v3 参考监督审核</h2><pre>'+html.escape(json.dumps(review,indent=2,ensure_ascii=False))+'</pre></article>')
    parts.append('<article><h2>正在测试</h2>')
    for version in ('sft_clean_mask_v3','sft_clean_mask_v4_raw','sft_attention_pilot_v1','sft_refinement_pilot_v1','sft_integer_pilot_v1','sft_extended_v1'):
        data=read(ROOT/'data'/version/'manifest.json')
        metrics=ROOT/'outputs'/version/'train/metrics.jsonl'
        latest=None
        if metrics.exists():
            lines=metrics.read_text().splitlines()
            if lines:
                try:latest=json.loads(lines[-1])
                except json.JSONDecodeError:pass
        total_steps={'sft_refinement_pilot_v1':1600,'sft_extended_v1':4000}.get(version,1200)
        status=(f"训练 step {latest['step']} / {total_steps}，最近训练 loss {latest['loss']:.4f}"if latest else'训练尚未启动；准备 / 审核监督中')
        if data:
            status+=f"；原生照片 {data['native_training_images']}"
            trajectories=data.get('editing_trajectories',data.get('trajectories'))
            if trajectories is not None:status+=f"，候选轨迹 {trajectories}"
        parts.append('<p>'+html.escape(version+'：'+status)+'</p>')
    parts.append('<p>每 300 步保存模型并运行固定八图的完整编辑。训练 loss 与图片编辑质量分别记录。</p></article>')
    rejudge=read(ROOT/'outputs/sft_clean_mask_v2/rejudge_delta_400/summary.json')
    if rejudge:
        parts.append('<article><h2>评估证据校验</h2><p>额外提供真实 RGB 差异给 judge，严格的完整目标 / 保留背景要求保持一致。七个控制中：三个真实轻微编辑通过，四个错误 / no-op 全部拒绝。</p><p>旧 checkpoint400 的同一组已保存图像重新评分：'+str(rejudge['strict_pass'])+'/'+str(rejudge['reviewed'])+'。没有重跑模型；原先 1/8 与这次分数属于不同的评估证据版本。</p></article>')
    records=[]
    for version in ('sft_clean_mask_v2','sft_clean_mask_v3','sft_clean_mask_v4_raw','sft_attention_pilot_v1','sft_refinement_pilot_v1','sft_integer_pilot_v1','sft_extended_v1'):
        for folder in sorted((ROOT/'outputs'/version).glob('development_*')):
            summary=read(folder/'summary.json')
            if not summary:continue
            records.append((version,folder,summary))
    parts.append('<article><h2>固定八图的实际 checkpoint 检查</h2><table><tr><th>模型</th><th>严格编辑通过</th><th>工具完成</th></tr>')
    for version,folder,summary in records:
        parts.append('<tr><td>'+html.escape(version+' / '+folder.name)+'</td><td>'+str(summary['judge_pass'])+'/'+str(summary['tasks'])+'</td><td>'+str(summary['complete'])+'/'+str(summary['tasks'])+'</td></tr>')
    parts.append('</table></article>')
    if records:
        version,folder,summary=records[-1]
        results=[json.loads(l)for l in(folder/'results.jsonl').open()]
        for result in results:
            case=Path(result['artifact_dir']);relative='../'+str(case.relative_to(ROOT/'outputs'))
            judgment=result.get('judge',{})
            parts.append('<article><h2>'+html.escape(result['id'])+' · '+('通过'if judgment.get('pass')else'未通过')+'</h2><div class="pair"><figure><figcaption>输入</figcaption><img src="'+relative+'/source.png"></figure><figure><figcaption>模型计算 RGB 的结果</figcaption><img src="'+relative+'/edited.png"></figure></div><p>'+html.escape(judgment.get('reason','尚未完成评分'))+'</p></article>')
    parts.append('<p>更新时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat()+'</p></html>')
    (out/'index.html').write_text(''.join(parts))
    print('LOCAL_SFT_PAGE_UPDATED',flush=True)


if __name__=='__main__':main()
