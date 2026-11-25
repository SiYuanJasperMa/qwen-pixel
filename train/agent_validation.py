"""Independent native editing validation without discarding the live optimizer."""
from contextlib import contextmanager
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def independent_workspace(task_folder):
    if task_folder.exists() and any(task_folder.iterdir()):
        return task_folder / ('retry_' + str(time.time_ns()))
    return task_folder


@contextmanager
def preserve_training_state(actor):
    modes = [(module, module.training) for module in actor.modules()]
    cache_setting = actor.config.use_cache
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        actor.eval()
        yield
    finally:
        actor.train(modes[0][1])
        for module, training in modes:
            module.training = training
            if hasattr(module, 'rope_deltas'):
                module.rope_deltas = None
        actor.config.use_cache = cache_setting
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def trainable_digest(actor):
    digest = hashlib.sha256()
    for name, value in actor.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(list(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluate(actor, processor, checkpoint, args):
    # Include lazy imports and client setup so they cannot perturb the training
    # sampler's RNG before the validation state is captured.
    with preserve_training_state(actor):
        return _evaluate(actor, processor, checkpoint, args)


def _evaluate(actor, processor, checkpoint, args):
    from scripts.run_agent_demo import rollout
    from pixel.agent_protocol import validate_outputs
    from pixel.decoding import DecodePolicyModel, decoding_identity
    from pixel.generation_cache import fresh_precision_cache,RecurrentPrecisionLayer
    from pixel.eval_manifest import digest, prepare_manifest, runtime_digest
    from pixel.teacher_api import judge_edit, make_client
    from pixel.region_preservation import load_registry,evaluate_region
    registry=load_registry(args.agent_val_tasks)
    tasks = [json.loads(s) for s in Path(args.agent_val_tasks).read_text().splitlines() if s.strip()]
    tasks = [task for task in tasks if task['split'] == args.agent_val_split]
    if args.agent_val_limit:
        tasks = tasks[:args.agent_val_limit]
    if not tasks:
        raise ValueError('Native editing validation has no tasks in the requested split.')
    out = checkpoint / 'agent_validation'
    weight_hash = trainable_digest(actor)
    configuration = {'checkpoint': str(checkpoint), 'trainable_weights_sha256': weight_hash,
        'numeric_settings': decoding_identity(args.agent_val_decode_policy),
        'profile': args.agent_val_profile, 'strict_extent': True,
        'max_steps': 8, 'max_new_tokens': 4096, 'loaded_rollout_sha256': runtime_digest(rollout),
        'validation_helper_sha256': digest(Path(__file__)),
        'loaded_validation_sha256': runtime_digest(_evaluate),
        'loaded_decoding_implementation':{
            'loaded_policy_sha256':runtime_digest(DecodePolicyModel.generate),
            'loaded_cache_factory_sha256':runtime_digest(fresh_precision_cache),
            'loaded_cache_initialization_sha256':runtime_digest(RecurrentPrecisionLayer.lazy_initialization)},
        'loaded_state_guard_sha256': runtime_digest(preserve_training_state.__wrapped__),
        'region_registry_sha256':digest(Path(args.agent_val_tasks).with_name('development_masks_eval_only.json'))if registry is not None else None,
        'loaded_region_evaluator_sha256':runtime_digest(evaluate_region),
        'region_evaluator_file_sha256':digest(ROOT/'pixel/region_preservation.py'),
        'inputs': [{'id':t['id'], 'instruction':t['instruction'], 'source_sha256':digest(t['image']),
                    'judge_target_box':t.get('target_bbox_pixels')} for t in tasks]}
    prepare_manifest(out, configuration)
    model = actor if args.agent_val_decode_policy == 'legacy' else DecodePolicyModel(actor, args.agent_val_decode_policy)
    results = []
    with preserve_training_state(actor):
        client = make_client()
        for task in tasks:
            task_folder = out / task['id']
            workspace = independent_workspace(task_folder)
            summary, _ = rollout(model, processor, task['image'], task['instruction'], workspace,
                max_steps=8, max_new_tokens=4096, model_name=args.model, adapter=str(checkpoint),
                system_profile=args.agent_val_profile, use_checkpoint_interface=True)
            result = {'id':task['id'], 'complete':summary['complete'], 'actions':summary['actions'],
                      'artifact_dir':str(workspace)}
            if summary['complete']:
                receipt = validate_outputs(workspace, summary['validation']['image_path'], summary['validation']['rgb_path'])
                if any(receipt[k] != summary['validation'][k] for k in ('image_sha256','rgb_sha256')):
                    raise ValueError('Completed native RGB changed before validation.')
                result['validation'] = receipt
                region=evaluate_region(task,workspace/'source.png',workspace/receipt['image_path'],registry)
                if region is not None:result['region_preservation']=region
                try:
                    result['judge'] = judge_edit(client, workspace/'source.png',
                        workspace/receipt['image_path'], task['instruction'], True, task.get('target_bbox_pixels'))
                except Exception as error:
                    result['judging_error'] = type(error).__name__
            (task_folder/'evaluation.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    after_hash = trainable_digest(actor)
    if after_hash != weight_hash:
        raise ValueError('Native validation modified trainable actor weights.')
    report = {'tasks':len(results), 'complete':sum(r['complete'] for r in results),
              'judged':sum('judge' in r for r in results),
              'judge_pass':sum(r.get('judge',{}).get('pass',False) for r in results),
              'decode_policy':args.agent_val_decode_policy,
              'trainable_weights_unchanged':True, 'training_modes_and_rng_restored':True,
              'training_split':args.agent_val_split, 'not_official_benchmark':True}
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    print('INLINE_AGENT_VALIDATION',json.dumps(report),flush=True)
    return report
