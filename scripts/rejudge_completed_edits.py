"""Review actual completed model images without regenerating or replacing scores."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pixel.agent_protocol import validate_outputs
from pixel.teacher_api import make_client,judge_edit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--tasks')
    args=parser.parse_args()
    source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks={t['id']:t for t in(json.loads(line)for line in Path(args.tasks).open())}if args.tasks else{}
    client=make_client();reviews=[]
    for path in sorted(source.glob('*/evaluation.json')):
        original=json.loads(path.read_text())
        if not original['complete']:continue
        destination=out/(original['id']+'.json')
        if destination.exists():reviews.append(json.loads(destination.read_text()));continue
        folder=Path(original.get('artifact_dir',path.parent))
        if not folder.is_absolute():folder=ROOT/folder
        trajectory=json.loads((folder/'trajectory.json').read_text())
        summary=json.loads((folder/'summary.json').read_text());validation=summary['validation']
        receipt=validate_outputs(folder,validation['image_path'],validation['rgb_path'])
        for key in ('image_sha256','rgb_sha256'):
            if key in validation and receipt[key]!=validation[key]:raise ValueError('Model image changed since completion.')
        review={'id':original['id'],'original_judge':original.get('judge'),
            'strict_judge':judge_edit(client,folder/'source.png',folder/validation['image_path'],trajectory['instruction'],True,
                tasks.get(original['id'],{}).get('target_bbox_pixels')),
            'image_sha256':receipt['image_sha256'],'actor_regenerated':False}
        destination.write_text(json.dumps(review,indent=2));reviews.append(review)
        print(json.dumps(review),flush=True)
    summary={'reviewed':len(reviews),'legacy_pass':sum(r.get('original_judge',{}).get('pass',False)for r in reviews),
        'strict_pass':sum(r['strict_judge'].get('pass',False)for r in reviews),
        'actor_regenerated':False,'rubric_version':'full_visible_target_v1'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('STRICT_REVIEW_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
