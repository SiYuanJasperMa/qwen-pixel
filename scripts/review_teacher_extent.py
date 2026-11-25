"""Check whether the existing SFT teacher targets satisfy their full instructions."""
import concurrent.futures
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import validate_outputs
from pixel.teacher_api import judge_edit,make_client


def review(task,out):
    path=out/(task['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    source=ROOT/'data/agent_rgb_teacher_v6'/task['id']
    receipt=validate_outputs(source)
    judgment=judge_edit(make_client(),source/'source.png',source/'edited.png',task['instruction'],True)
    result={'id':task['id'],'instruction':task['instruction'],'strict_judge':judgment,
            'image_sha256':receipt['image_sha256'],'new_teacher_generation':False}
    path.write_text(json.dumps(result,indent=2))
    return result


def main():
    out=ROOT/'outputs/sft_audit/teacher_extent';out.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(l)for l in(ROOT/'data/sft_seven_tasks_v1/tasks.jsonl').open()]
    tasks=[t for t in tasks if t['split']=='train']
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=7)as pool:
        futures=[pool.submit(review,t,out)for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    summary={'teachers':len(results),'strict_pass':sum(r['strict_judge']['pass']for r in results),
             'new_teacher_generation':False,'rubric_version':'full_visible_target_v1'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('TEACHER_EXTENT_REVIEW',json.dumps(summary),flush=True)


if __name__=='__main__':main()
