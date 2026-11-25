"""Separate completed executions, pending judgments and semantic failure causes."""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(folder):
    paths=sorted(folder.glob('*/evaluation.json'))
    counts=Counter();cases=[]
    for path in paths:
        row=json.loads(path.read_text());counts['observed']+=1
        if row.get('complete'):counts['complete']+=1
        work=Path(row.get('artifact_dir',path.parent))
        trajectory=work/'trajectory.json'
        actions=[];repeated_identical_calls=0
        if trajectory.is_file():
            turns=json.loads(trajectory.read_text()).get('turns',[])
            actions=[t.get('action')for t in turns]
            signatures=[json.dumps({'action':t.get('action'),'arguments':t.get('arguments')},sort_keys=True)
                        for t in turns if t.get('action')in ('python','view_image','finish')]
            repeated_identical_calls=len(signatures)-len(set(signatures))
            if repeated_identical_calls:counts['repeated_identical_call_cases']+=1
            if 'parse_error'in actions:counts['parse_error_cases']+=1
            if 'output_budget_exceeded'in actions:counts['truncated_cases']+=1
            if 'context_limit'in actions:counts['context_limit_cases']+=1
            if any(t.get('action')=='python'and t.get('passed')is False for t in turns):counts['python_error_cases']+=1
        judge=row.get('judge')
        causes=[]
        if not row.get('complete'):causes.append('execution_incomplete')
        elif not judge:
            causes.append('judge_pending');counts['judge_pending']+=1
        else:
            counts['judged']+=1
            if judge.get('pass')is True:counts['semantic_pass']+=1
            else:
                for field,minimum,cause in [('instruction_adherence',4,'instruction'),('preservation',4,'preservation'),('visual_quality',3,'visual_quality'),('target_extent_score',4,'target_extent')]:
                    if field in judge and judge[field]<minimum:causes.append(cause);counts[cause+'_fail']+=1
                if judge.get('target_extent_complete')is False and'target_extent'not in causes:
                    causes.append('target_extent');counts['target_extent_fail']+=1
                if not causes:causes.append('judge_rejected')
        cases.append({'id':row['id'],'complete':bool(row.get('complete')),'semantic_pass':judge.get('pass')if judge else None,
                      'causes':causes,'actions':actions,'repeated_identical_calls':repeated_identical_calls,
                      'reason':judge.get('reason')if judge else row.get('judging_error')})
    return {'counts':dict(counts),'causes_can_overlap':True,'cases':cases,
            'criterion':'actual native RGB completion and full-visible-target judge, reported separately'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eval',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();report=summarize(args.eval)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['counts']))


if __name__=='__main__':main()
