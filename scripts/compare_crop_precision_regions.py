"""Measure saved actual RGB for precision-policy comparisons without regenerating it."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import validate_outputs
from pixel.region_preservation import load_registry,evaluate_region


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    tasks_path=ROOT/'data/agent_sft_en_27b_37photos_20261125/development_tasks.jsonl'
    tasks={r['id']:r for r in map(json.loads,tasks_path.read_text().splitlines())}
    registry=load_registry(tasks_path)
    variants={'deterministic':ROOT/'outputs/agent_sft_crop_feedback_paired_20261125/train/checkpoint-480/agent_validation',
              'fp32_state':ROOT/'outputs/agent_sft_crop_fp32_development_20261125',
              'untuned_fp32_state':ROOT/'outputs/agent_sft_native_base_fp32_development_20261125',
              'feedback_parent_fp32':ROOT/'outputs/agent_sft_feedback_parent_fp32_development_20261125',
              'feedback450_fp32':ROOT/'outputs/agent_sft_feedback450_fp32_development_20261125'}
    variants['feedback450_deterministic']=ROOT/'outputs/agent_sft_native_feedback_reproduction_20261125/train/checkpoint-450/agent_validation'
    variants['control450_deterministic']=ROOT/'outputs/agent_sft_native_feedback_control_20261125/train/checkpoint-450/agent_validation'
    results={}
    for policy,folder in variants.items():
        rows=[]
        for tid,task in tasks.items():
            p=folder/tid/'evaluation.json'
            if not p.exists():continue
            evaluation=json.loads(p.read_text())
            case={'id':tid,'complete':evaluation.get('complete',False),'judge_pass':(evaluation.get('judge')or{}).get('pass')}
            if case['complete']:
                workspace=Path(evaluation.get('artifact_dir',folder/tid))
                receipt=validate_outputs(workspace)
                region=evaluate_region(task,workspace/'source.png',workspace/receipt['image_path'],registry)
                case['region']=region
                case['actual_rgb_sha256']=receipt['rgb_sha256']
                # A crop that omits the target bounds any program that clips
                # its edit to that crop. This is distinct from mask precision.
                box_file=workspace/'region.json'
                if box_file.exists():
                    box=json.loads(box_file.read_text())['source_bbox_pixels']
                    entry=registry['instances'][tid]
                    with Image.open(entry['mask'])as image:mask=np.asarray(image.convert('L'))>0
                    x0,y0,x1,y1=box
                    inside=mask[max(0,y0):min(mask.shape[0],y1),max(0,x0):min(mask.shape[1],x1)]
                    case['selected_source_box']=box
                    case['target_pixels_inside_selected_box_fraction']=float(inside.sum()/mask.sum())
            rows.append(case)
        results[policy]={'planned':len(tasks),'observed':len(rows),'cases':rows,
            'scope':'immutable saved RGB and target registry; untuned baseline has different weights',
            'not_official_or_semantic_score':True}
    args.out.write_text(json.dumps(results,indent=2))
    print(json.dumps({k:{'observed':v['observed'],'complete':sum(r['complete']for r in v['cases']),
        'judged':sum(r['judge_pass']is not None for r in v['cases'])}for k,v in results.items()}))


if __name__=='__main__':main()
