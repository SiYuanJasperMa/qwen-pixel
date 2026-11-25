"""Evaluate autonomous Qwen trajectories on image-disjoint pilot tasks."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from run_agent_demo import rollout
from run_pixel_sft import load
from pixel.teacher_api import judge_edit, make_client
from pixel.eval_manifest import digest,prepare_manifest,runtime_digest
from pixel.agent_protocol import execute_action,validate_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--max-new-tokens', type=int, default=1800)
    parser.add_argument('--system-profile', choices=['default', 'grounded', 'rgb', 'roi','rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'], default='default')
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tool-image-mode', choices=['tool','user'], default='tool')
    parser.add_argument('--strict-extent',action='store_true')
    parser.add_argument('--checkpoint-interface',action='store_true')
    parser.add_argument('--decode-policy',choices=['legacy','deterministic','fp32_state'],default='legacy')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluation_lock=(out/'.evaluation.lock').open('a')
    fcntl.flock(evaluation_lock.fileno(),fcntl.LOCK_EX)
    tasks = [json.loads(l) for l in Path(args.tasks).open()]
    tasks = [t for t in tasks if t['split'] == args.split]
    if args.limit:
        tasks = tasks[:args.limit]
    if any(t.get('strict_target_extent')for t in tasks):args.strict_extent=True
    configuration=vars(args).copy();configuration.pop('out')
    if args.decode_policy=='legacy':
        # Preserve the frozen legacy identity for already-running comparisons.
        configuration.pop('decode_policy')
    else:
        from pixel.decoding import DecodePolicyModel,configure_decoding,decoding_identity
        from pixel.generation_cache import fresh_precision_cache,RecurrentPrecisionLayer
        configure_decoding(args.decode_policy)
        configuration['numeric_settings']=decoding_identity(args.decode_policy)
        configuration['decoding_policy_sha256']=digest(ROOT/'pixel/decoding.py')
        configuration['recurrent_cache_sha256']=digest(ROOT/'pixel/generation_cache.py')
        configuration['loaded_policy_sha256']=runtime_digest(DecodePolicyModel.generate)
        configuration['loaded_cache_factory_sha256']=runtime_digest(fresh_precision_cache)
        configuration['loaded_cache_initialization_sha256']=runtime_digest(RecurrentPrecisionLayer.lazy_initialization)
    configuration['adaptive_output_ceiling']=6144
    configuration['inputs']=[{'id':t['id'],'instruction':t['instruction'],
        'source_sha256':digest(t['image']),'judge_target_box':t.get('target_bbox_pixels')}for t in tasks]
    configuration['judge_protocol']='annotated_instance_crop_and_aligned_rgb_delta_v2'
    configuration['difference_evidence_sha256']=digest(ROOT/'pixel/edit_difference.py')
    configuration['runner_sha256']=digest(ROOT/'scripts/run_agent_demo.py')
    configuration['environment_sha256']=digest(ROOT/'pixel/agent_protocol.py')
    configuration['loaded_runner_sha256']=runtime_digest(rollout)
    configuration['loaded_environment_sha256']=runtime_digest(execute_action)
    configuration['loaded_judge_sha256']=runtime_digest(judge_edit)
    if args.adapter:
        weights=Path(args.adapter)/'adapter_model.safetensors'
        configuration['adapter_sha256']=digest(weights)
    prepare_manifest(out,configuration)
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=512 * 512))
    if args.decode_policy!='legacy':model=DecodePolicyModel(model,args.decode_policy)
    __import__('torch').manual_seed(args.seed)
    client = make_client()
    results = []
    for task in tasks:
        workspace = out / task['id']
        result_file = workspace / 'evaluation.json'
        if result_file.exists():
            existing=json.loads(result_file.read_text())
            if not existing['complete']or 'judge'in existing:
                results.append(existing)
                continue
        artifact_dir=Path(existing.get('artifact_dir',workspace))if result_file.exists()else workspace
        summary_file=artifact_dir/'summary.json'
        if summary_file.exists():
            summary=json.loads(summary_file.read_text())
        else:
            # A interrupted model run is not valid cached output. Preserve it
            # and make the new execution independent of any stale edited file.
            if (artifact_dir/'trajectory.json').exists():
                artifact_dir=workspace/f'retry_{time.time_ns()}'
            summary, _ = rollout(model, processor, task['image'], task['instruction'],
                artifact_dir, args.max_steps, args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                model_name=args.model, adapter=args.adapter, system_profile=args.system_profile, tool_image_mode=args.tool_image_mode,use_checkpoint_interface=args.checkpoint_interface)
        result = {'id': task['id'], 'complete': summary['complete'], 'steps': summary['steps'],
                  'artifact_dir':str(artifact_dir)}
        # Preserve model completion before contacting the judge. A transient
        # evaluator failure must not force a new model generation on resume.
        result_file.parent.mkdir(parents=True,exist_ok=True)
        result_file.write_text(json.dumps(result,indent=2))
        if summary['complete']:
            try:
                validation=summary['validation']
                receipt=validate_outputs(artifact_dir,validation['image_path'],validation['rgb_path'])
                for key in ('image_sha256','rgb_sha256'):
                    if key in validation and receipt[key]!=validation[key]:
                        raise ValueError('Completed output changed before evaluation.')
                result['judge'] = judge_edit(client, artifact_dir / 'source.png',
                    artifact_dir / validation['image_path'], task['instruction'],args.strict_extent,
                    task.get('target_bbox_pixels'))
            except Exception as error:
                result['judging_error']=type(error).__name__
        result_file.write_text(json.dumps(result, indent=2))
        results.append(result)
        print('EVAL_TASK', json.dumps(result), flush=True)
        (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    summary = {'tasks': len(results), 'complete': sum(r['complete'] for r in results),
        'judge_pass': sum(r.get('judge', {}).get('pass', False) for r in results),
        'split': args.split, 'model': args.model, 'adapter': args.adapter,
        'max_steps':args.max_steps,'initial_max_new_tokens':args.max_new_tokens,
        'adaptive_output_ceiling':6144,
        'strict_extent':args.strict_extent,
        'decode_policy':args.decode_policy,
        'judging_errors':sum('judging_error'in r for r in results),
        'note': ('Image-disjoint pilot holdout from COCO photos; not an official benchmark score.'
                 if args.split == 'val' else 'Training-task independent replay; not a held-out result.')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('AGENT_EVAL_COMPLETE', json.dumps(summary), flush=True)
    evaluation_lock.close()


if __name__ == '__main__':
    main()
