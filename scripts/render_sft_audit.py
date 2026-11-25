"""Show SFT correctness evidence and actual RGB edits in one local page."""
import base64
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sft_audit'


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def picture(path):
    path=Path(path)
    if not path.exists():return '<p>等待实际输出</p>'
    return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def main():
    audit=read(OUT/'consistency.json',[])
    simplified=[{'id':x['id'],'训练 loss':round(x['reported_loss'],6),
        '手工核对 loss':round(x['manual_shifted_loss'],6),
        'cache 开关结果一致':x['generated']['True']==x['generated']['False']}for x in audit]
    processor_parity=read(OUT/'processor_parity.json',{})
    strict_review=read(OUT/'strict_normalized_review/summary.json',{'状态':'等待完整目标复核'})
    independent=read(ROOT/'data/independent_coco_edit_eval_v2/manifest.json',{})
    single=ROOT/'outputs/sft_exact_replay_v1'
    prefixes=read(single/'prefixes.json',[])
    actions=[{'expected':x['expected'],'predicted':x['predicted']}for x in prefixes]
    case=single/'rollout/coco_agent_008'
    pixels=read(case/'pixel_check.json',{})
    checks=[]
    for split in ('train_rollouts','val_rollouts'):
        folder=ROOT/'outputs/sft_three_tasks_v1'/split
        checks.append({'split':split,'summary':read(folder/'summary.json',{'running':True})})
    experiments=[]
    for name,label in [('sft_seven_tasks_v1','七条原始轨迹'),
                       ('sft_checked_v1','数值错误反馈与修复'),
                       ('sft_normalized_v1','明确坐标单位与修复'),
                       ('sft_roi_v1','先定位并读取区域，再写编辑程序')]:
        folder=ROOT/'outputs'/name
        metrics=folder/'train/metrics.jsonl'
        latest=None
        if metrics.exists():
            lines=[line for line in metrics.read_text().splitlines()if line.strip()]
            if lines:latest=json.loads(lines[-1])
        prefix=read(folder/'prefixes.json',[])
        holdout=[read(path)for path in (folder/'val_rollouts').glob('coco_agent_*/evaluation.json')]
        training=[read(path)for path in (folder/'train_rollouts').glob('coco_agent_*/evaluation.json')]
        experiments.append({'实验':label,'训练步':latest['step']if latest else'等待训练',
            '已核对调用':len(prefix),'完整调用正确':sum(x['expected']==x['predicted']and x.get('complete_call',False)for x in prefix),
            '参数完全一致':sum(x.get('arguments_match',False)for x in prefix),
            '训练内有效提交':sum(x['complete']for x in training),
            '训练内编辑通过':sum(x.get('judge',{}).get('pass',False)for x in training),
            '留出图已测':len(holdout),'留出图有效提交':sum(x['complete']for x in holdout),
            '留出图编辑通过':sum(x.get('judge',{}).get('pass',False)for x in holdout)})
    examples=[]
    for split in ('train_rollouts','val_rollouts'):
        for folder in sorted((ROOT/'outputs/sft_three_tasks_v1'/split).glob('coco_agent_*')):
            trajectory=read(folder/'trajectory.json')
            if not trajectory:continue
            examples.append('<article><h3>'+html.escape(split+' · '+folder.name)+'</h3><p>'
                +html.escape(trajectory['instruction'])+'</p><div class="pair">'
                +picture(folder/'source.png')+picture(folder/'edited.png')+'</div><pre>'
                +html.escape(json.dumps(read(folder/'evaluation.json',{}),ensure_ascii=False,indent=2))+'</pre></article>')
    for name in ('sft_seven_tasks_v1','sft_checked_v1','sft_normalized_v1','sft_roi_v1'):
        for folder in sorted((ROOT/'outputs'/name/'val_rollouts').glob('coco_agent_*')):
            trajectory=read(folder/'trajectory.json')
            if not trajectory:continue
            evaluation=read(folder/'evaluation.json',{'状态':'正在执行，尚未完成判图'})
            examples.append('<article><h3>'+html.escape(name+' · '+folder.name)
                +'</h3><p>'+html.escape(trajectory['instruction'])+'</p><div class="pair">'
                +picture(folder/'source.png')+picture(folder/'edited.png')+'</div><pre>'
                +html.escape(json.dumps(evaluation,ensure_ascii=False,indent=2))+'</pre></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Pixel · SFT 正确性验证</title><style>
body{font-family:system-ui,sans-serif;max-width:1050px;margin:auto;padding:26px;background:#f4f7fb;color:#172333}article{background:white;padding:20px;border-radius:12px;margin:20px 0}p{line-height:1.7}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f9;padding:15px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<h1>Qwen Pixel · SFT 正确性验证</h1><p>先验证真实轨迹能否被学习和独立执行，再评估未训练图片。当前只做 SFT，暂不扩大 RL 或造数。</p>
<article><h2>训练与推理核对</h2><p>同一初始输入和工具定义逐字一致；结构化教师调用与模型调用序列化一致；手工移位交叉熵与训练 loss 一致；开关生成缓存输出一致。新增 checkpoint 保存并校验 system、tools、图像角色及图像预算。</p><pre>'''+html.escape(json.dumps(simplified,ensure_ascii=False,indent=2))+'''</pre></article>
<article><h2>单条真实轨迹 · 120 步 SFT</h2><p>任务：使左侧屏幕更亮。输入与输出保留原生尺寸。该样本用于过拟合诊断，不能代表泛化能力。</p><div class="pair">'''+picture(case/'source.png')+picture(case/'edited.png')+'''</div><pre>'''+html.escape(json.dumps({'下一步操作':actions,'像素核对':pixels,'完整执行':read(case/'summary.json',{})},ensure_ascii=False,indent=2))+'''</pre></article><article><h2>三任务与留出图片</h2><pre>'''+html.escape(json.dumps(checks,ensure_ascii=False,indent=2))+'''</pre></article><article><h2>正在核对的 SFT 形式</h2><p>这些小实验使用已有训练图片和相同编辑任务。错误反馈来自实际执行；修复样本不监督故意错误的调用。规范化程序重执行后与教师像素完全一致。未完成核对不能算成功，新图提交有效文件也不能算编辑正确。表中旧判图分数不能证明完整目标已被编辑，必须结合下面的严格复核。</p><pre>'''+html.escape(json.dumps(experiments,ensure_ascii=False,indent=2))+'''</pre></article><article><h2>实际张量与完整目标检查</h2><pre>'''+html.escape(json.dumps({'原始与保存processor张量完全一致':processor_parity.get('all_tensors_identical'),'实际核对样本数':len(processor_parity.get('examples',[])),'严格目标覆盖复核':strict_review,'独立检查集':{'图像数':independent.get('images'),'split':independent.get('official_image_split'),'只用于评测':independent.get('evaluation_only'),'任务指纹':independent.get('tasks_sha256')}},ensure_ascii=False,indent=2))+'''</pre><p>独立检查集在模型评分前固定，不能用于 SFT。只有脸部模糊或小块区域变亮，不得当作整个目标编辑成功。</p></article>'''+''.join(examples)+'</html>'
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')


if __name__=='__main__':main()
