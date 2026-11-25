"""Finish only missing image evaluations; never generate or repair teacher code."""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.teacher_api import make_client,judge_edit
from pixel.rgb_text_eval import pixel_metrics
from pixel.persistent_io import atomic_text


def finish(path):
    r=json.loads(path.read_text())
    if not r.get('teacher_usage')or r.get('execution',{}).get('status')!='passed':return
    source=Path(r['source_image']);target=Path(r['target_image']);edited=Path(r['execution']['output'])
    with Image.open(source)as image:source_size=image.size
    with Image.open(edited)as image:output_size=image.size
    r['output_resolution']=list(output_size)
    judge_source=source
    if source_size!=output_size:
        # Normalize only evaluator views. Keep the emitted code and edited.png unchanged.
        integer_enlargement=(output_size[0]%source_size[0]==0 and output_size[1]%source_size[1]==0
            and output_size[0]//source_size[0]==output_size[1]//source_size[1])
        sampling=Image.Resampling.NEAREST if integer_enlargement else Image.Resampling.LANCZOS
        judge_source=path.parent/'source_for_judge.png'
        with Image.open(source)as image:image.convert('RGB').resize(output_size,sampling).save(judge_source)
        aligned=path.parent/'edited_for_pixel_metrics.png'
        with Image.open(edited)as image:image.convert('RGB').resize(source_size,Image.Resampling.LANCZOS).save(aligned)
        r['metrics']=pixel_metrics(source,target,aligned)
        r['metrics']['metric_protocol']='aligned_resized_rgb_smoke_v1_output_resized_to_source'
        r['evaluation_alignment']={'source_input_size':list(source_size),'actual_output_size':list(output_size),
            'judge_source_resampling':'nearest'if integer_enlargement else'lanczos',
            'pixel_metric_output_resampling':'lanczos','code_or_output_changed':False}
    elif not r.get('metrics'):
        r['metrics']=pixel_metrics(source,target,edited)
    if not r.get('judge')or r.get('judge_pending'):
        try:
            r['judge']=judge_edit(make_client(),judge_source,edited,r['instruction'],strict_extent=True)
            r.pop('judge_pending',None);r.pop('judge_error',None)
            if 'error'in r:r['prior_evaluation_error']=r.pop('error')
            print('JUDGED',r['id'],r['mode'],r['judge']['pass'],flush=True)
        except Exception as exc:
            r['judge_pending']=True;r['judge_error']=type(exc).__name__+': '+str(exc)[:180]
            print('JUDGE_PENDING',r['id'],r['mode'],type(exc).__name__,flush=True)
    atomic_text(path,json.dumps(r,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'outputs/gpt61_rgb_gain_val64_v1'))
    args=parser.parse_args();folder=Path(args.out)
    paths=list(folder.glob('*/*/record.json'))
    with cf.ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(finish,paths))
    records=[json.loads(p.read_text())for p in sorted(paths)]
    atomic_text(folder/'results.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in records))


if __name__=='__main__':main()
