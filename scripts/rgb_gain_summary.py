"""Paired outcome statistics; keep API failures separate from edit failures."""
import json
import math
import random
from pathlib import Path


def paired_stats(pairs):
    wins=sum(p['rgb_pass']and not p['image_pass']for p in pairs)
    losses=sum(p['image_pass']and not p['rgb_pass']for p in pairs)
    n=len(pairs);discordant=wins+losses
    pvalue=min(1.,2*sum(math.comb(discordant,k)for k in range(min(wins,losses)+1))/2**discordant)if discordant else 1.
    interval=None
    if n:
        diffs=[int(p['rgb_pass'])-int(p['image_pass'])for p in pairs]
        rng=random.Random(20261124)
        draws=sorted(100*sum(rng.choices(diffs,k=n))/n for _ in range(5000))
        interval=[draws[125],draws[4874]]
    return {'paired_evaluated':n,'image_only_passed':sum(p['image_pass']for p in pairs),
            'image_rgb_passed':sum(p['rgb_pass']for p in pairs),
            'rgb_only_wins':wins,'image_only_wins':losses,
            'net_gain_pp':100*(wins-losses)/n if n else None,'mcnemar_exact_p':pvalue if n else None,
            'paired_bootstrap_95ci_pp':interval}


def collect(folder):
    records={}
    for path in folder.glob('*/*/record.json'):
        try:
            r=json.loads(path.read_text());records[(r['id'],r['mode'])]=r
        except json.JSONDecodeError:pass
    modes={}
    def pending(r):
        return bool(r.get('judge_pending')or(r.get('execution',{}).get('status')=='passed'and not r.get('judge')))
    for mode in ['image_only','image_rgb']:
        rows=[r for(i,m),r in records.items()if m==mode]
        valid=[r for r in rows if r.get('teacher_usage')and not pending(r)]
        modes[mode]={'completed':len(rows),'expected':64,'api_failed':sum(not r.get('teacher_usage')for r in rows),
                     'judge_pending':sum(pending(r)for r in rows),
                     'evaluated':len(valid),'passed':sum(r.get('judge',{}).get('pass')is True for r in valid),
                     'executed':sum(r.get('execution',{}).get('status')=='passed'for r in rows)}
        modes[mode]['format_or_code_failures']=sum(r.get('execution',{}).get('status')!='passed'for r in valid)
        modes[mode]['semantic_failures']=sum(r.get('execution',{}).get('status')=='passed'and r.get('judge',{}).get('pass')is not True for r in valid)
        usage=[r['teacher_usage']for r in rows if r.get('teacher_usage')]
        if usage:
            modes[mode].update(mean_input_tokens=sum(u['input_tokens']for u in usage)/len(usage),
                               mean_output_tokens=sum(u['output_tokens']for u in usage)/len(usage),
                               total_input_tokens=sum(u['input_tokens']for u in usage),
                               total_output_tokens=sum(u['output_tokens']for u in usage),
                               mean_reasoning_tokens=sum((u.get('output_tokens_details')or{}).get('reasoning_tokens',0)for u in usage)/len(usage))
        metrics=[r['metrics']for r in valid if r.get('metrics')]
        if metrics:
            modes[mode]['pixel_metrics']={'evaluated':len(metrics),'mean_l1':sum(m['l1']for m in metrics)/len(metrics),
                'mean_no_op_l1':sum(m['no_op_l1']for m in metrics)/len(metrics),
                'better_than_no_op':sum(m['l1']<m['no_op_l1']for m in metrics),
                'protocol':'local aligned-resized RGB L1; not official benchmark evaluation'}
    pairs=[]
    for tid in sorted({i for i,m in records}):
        a,b=records.get((tid,'image_only')),records.get((tid,'image_rgb'))
        if not a or not b or not a.get('teacher_usage')or not b.get('teacher_usage')or pending(a)or pending(b):continue
        assert a['source_rgb_sha256']==b['source_rgb_sha256']
        pairs.append({'id':tid,'dataset':a['dataset'],'image_pass':a.get('judge',{}).get('pass')is True,'rgb_pass':b.get('judge',{}).get('pass')is True})
    by_dataset={ds:paired_stats([p for p in pairs if p['dataset']==ds])for ds in sorted({p['dataset']for p in pairs})}
    judged_pairs=[(records[(p['id'],'image_only')],records[(p['id'],'image_rgb')])for p in pairs
                  if all(records[(p['id'],m)].get('judge')for m in ['image_only','image_rgb'])]
    quality={}
    if judged_pairs:
        for i,m in enumerate(['image_only','image_rgb']):
            quality[m]={key:sum(pair[i]['judge'].get(key,0)for pair in judged_pairs)/len(judged_pairs)
                        for key in ['instruction_adherence','preservation','visual_quality']}
        quality['paired_judged']=len(judged_pairs)
    metric_pairs=[(records[(p['id'],'image_only')],records[(p['id'],'image_rgb')])for p in pairs
                  if all(records[(p['id'],m)].get('metrics')for m in ['image_only','image_rgb'])]
    paired_pixels={}
    if metric_pairs:
        paired_pixels={'paired_measured':len(metric_pairs),'protocol':'local RGB L1; generated output resized to source only if dimensions differ'}
        for i,m in enumerate(['image_only','image_rgb']):
            paired_pixels[m]={'mean_l1':sum(pair[i]['metrics']['l1']for pair in metric_pairs)/len(metric_pairs),
                             'better_than_no_op':sum(pair[i]['metrics']['l1']<pair[i]['metrics']['no_op_l1']for pair in metric_pairs)}
        paired_pixels['mean_no_op_l1']=sum(pair[0]['metrics']['no_op_l1']for pair in metric_pairs)/len(metric_pairs)
    token_comparison={}
    if all('mean_input_tokens'in m for m in modes.values()):
        a,b=modes['image_only'],modes['image_rgb']
        token_comparison={'extra_mean_input_tokens':b['mean_input_tokens']-a['mean_input_tokens'],
                          'extra_mean_output_tokens':b['mean_output_tokens']-a['mean_output_tokens'],
                          'cache_discount_assumed':False,
                          'price_note':'No verified account prices; image tokens/fees may not be fully exposed by proxy usage.'}
    return {'modes':modes,**paired_stats(pairs),'by_dataset':by_dataset,'token_comparison':token_comparison,
            'paired_quality_scores':quality,'paired_pixel_metrics':paired_pixels,
            'judge_model':'gpt-5.5-2026-04-24','resized_output_cases':[{'id':r['id'],'mode':r['mode'],**r['evaluation_alignment']}
                for r in records.values()if r.get('evaluation_alignment')],
            'complete':all(m['completed']==64 and m['api_failed']==0 and m['judge_pending']==0 for m in modes.values()),
            'pairs':pairs,'criterion':'custom VLM judge of source + edited image + instruction, full edit-object extent; no ground-truth image supplied to judge',
            'resolution':'128x128 resized','confidence_interval_method':'paired nonparametric bootstrap, 5000 resamples, fixed seed'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    folder=root/'outputs/gpt61_rgb_gain_val64_v1';summary=collect(folder)
    (folder/'paired_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items()if k!='pairs'},ensure_ascii=False))
