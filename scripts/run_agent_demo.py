"""Let Qwen choose and execute a sequence of pixel editing actions."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from pixel.agent_protocol import (TOOLS, RGB_TOOLS, RGB_SYSTEM, CHECKED_RGB_TOOLS, CHECKED_RGB_SYSTEM, NORMALIZED_RGB_SYSTEM, REGION_EDIT_TOOLS, REGION_EDIT_SYSTEM, add_region_guidance, ROI_TOOLS, ROI_SYSTEM, GROUNDED_GUIDANCE, assistant_call, copy_source_image, execute_action,
                                  export_sft_steps, initial_messages, parse_call)
from run_pixel_sft import encode, load
from pixel.sft_contract import validate_runtime_contract,checkpoint_interface
from pixel.eval_manifest import output_budgets


def rollout_export_identity(messages, model_name='', adapter=''):
    payload = json.dumps({'messages': messages, 'model': model_name, 'adapter': adapter},
                         sort_keys=True, separators=(',', ':')).encode()
    return 'qwen_rollout_'+hashlib.sha256(payload).hexdigest()[:24]


def export_rollout_sft(messages, samples, output, model_name, adapter, tools, stop_id):
    assistants = [(i, m) for i, m in enumerate(messages) if m['role'] == 'assistant']
    if len(assistants) != len(samples):
        raise ValueError('Sampled actions do not match exported assistant history.')
    for (index, message), sample in zip(assistants, samples):
        ids = sample.get('target_token_ids', [])
        if not ids or ids[-1] != stop_id:
            raise ValueError('Only complete sampled assistant actions may be exported.')
        if sample['messages'] != messages[:index] or sample['target_text'] != message['content']:
            raise ValueError('Sampled action differs from its native visible history.')
    identity = rollout_export_identity(messages, model_name, adapter)
    export_sft_steps(messages, output, identity,
                     'actual_qwen_rollout; visual quality not yet judged', tools)
    rows = [json.loads(line) for line in Path(output).read_text().splitlines()]
    for row, sample in zip(rows, samples):
        row['target_token_ids'] = sample['target_token_ids']
        row['tool_image_mode'] = sample.get('tool_image_mode', 'tool')
        row['meta']['sampled_token_policy'] = 'exact_complete_native_action_through_eos'
    Path(output).write_text(''.join(json.dumps(row)+'\n' for row in rows))


def rollout(model, processor, source, instruction, out, max_steps=8,
            max_new_tokens=1800, temperature=0.0, max_context=8192, model_name='', adapter='', system_profile='default', top_p=None, tool_image_mode='tool',use_checkpoint_interface=False,resume_prefix=None):
    out = Path(out).resolve()
    if resume_prefix is not None and out.exists()and any(out.iterdir()):
        raise ValueError('Prefix continuation requires a fresh diagnostic workspace.')
    out.mkdir(parents=True, exist_ok=True)
    copy_source_image(source, out / 'source.png')
    model.eval()
    messages = initial_messages(out / 'source.png', instruction)
    if system_profile == 'grounded':
        messages[0]['content'] += GROUNDED_GUIDANCE
    if system_profile == 'rgb':
        messages[0]['content'] = RGB_SYSTEM
    if system_profile == 'rgb_checked':
        messages[0]['content'] = CHECKED_RGB_SYSTEM
    if system_profile == 'rgb_normalized':
        messages[0]['content'] = NORMALIZED_RGB_SYSTEM
    if system_profile == 'roi_edit':
        messages[0]['content'] = REGION_EDIT_SYSTEM
        add_region_guidance(messages)
    if system_profile == 'clean_edit':
        from pixel.clean_edit_protocol import initial as clean_initial
        messages=clean_initial(out/'source.png',instruction)
    if system_profile == 'integer_edit':
        from pixel.integer_edit_protocol import initial as integer_initial
        messages=integer_initial(out/'source.png',instruction)
    if system_profile == 'roi':
        messages[0]['content'] = ROI_SYSTEM
    tools = (REGION_EDIT_TOOLS if system_profile == 'roi_edit'else CHECKED_RGB_TOOLS
             if system_profile in ('rgb_checked','rgb_normalized') else ROI_TOOLS
             if system_profile == 'roi' else RGB_TOOLS if system_profile == 'rgb' else TOOLS)
    if system_profile == 'clean_edit':
        from pixel.clean_edit_protocol import TOOLS as clean_tools
        tools=clean_tools
    if system_profile == 'integer_edit':
        from pixel.integer_edit_protocol import TOOLS as integer_tools
        tools=integer_tools
    image_size = getattr(processor.image_processor, 'size', {})
    runtime_features={'rgb':system_profile in ('rgb','roi','rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'),
        'roi':system_profile in ('roi','roi_edit','clean_edit','integer_edit'),
        'checked_rgb':system_profile in ('rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'),
        'python_scope':'per_call','files_persist':True}
    restored_history=None
    if resume_prefix is not None:
        if system_profile!='clean_edit':raise ValueError('Verified prefix continuation currently requires clean_edit.')
        if resume_prefix['messages'][0]!=messages[0]or resume_prefix['tools']!=tools:
            raise ValueError('Supplied prefix differs from requested runtime interface.')
        prefix_instruction=next(m.get('edit_instruction')for m in resume_prefix['messages']if m['role']=='user')
        if prefix_instruction!=instruction:raise ValueError('Prefix instruction mismatch.')
        from pixel.prefix_replay import replay_prefix
        restored_history=replay_prefix(resume_prefix,out)
        messages=restored_history.pop('rebound_messages')
    if use_checkpoint_interface:
        messages,tools=checkpoint_interface(adapter,messages,tools,runtime_features)
    validate_runtime_contract(adapter, messages[0]['content'], tools, tool_image_mode,
                              image_size.get('longest_edge', 262144),messages,runtime_features)
    turns = []
    training_rows = []
    complete = False
    for step in range(max_steps):
        row = {'id': f'rollout_{step}', 'messages': messages, 'tools': tools, 'tool_image_mode': tool_image_mode}
        inputs = encode(processor, row, model.device)
        budgets=output_budgets(int(inputs['input_ids'].shape[1]),max_new_tokens,max_context)
        if temperature>0:
            # Stochastic training rollouts must not silently resample until a
            # completion is accepted. The SFT replay checks use greedy decoding.
            budgets=budgets[:1]
        if not budgets:
            turns.append({'step': step, 'action': 'context_limit'})
            break
        start = time.time()
        sample_args = {'temperature': temperature, 'top_p': .95 if top_p is None else top_p, 'top_k': 20} if temperature > 0 else {}
        attempts=[]
        stop_id=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        for budget in budgets:
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=budget,
                    do_sample=temperature > 0, use_cache=True, **sample_args,
                    eos_token_id=stop_id,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            tokens = generated[0, inputs['input_ids'].shape[1]:]
            stopped=len(tokens)>0 and int(tokens[-1])==stop_id
            attempts.append({'budget':budget,'output_tokens':len(tokens),'complete':stopped})
            if stopped:break
        text = processor.tokenizer.decode(tokens, skip_special_tokens=False)
        text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
        # A budget-truncated action has no sampled EOS: do not invent one for RL.
        if stopped:
            training_rows.append({'id': f'rollout_{step}', 'task': 'agent_edit',
                'messages': list(messages), 'tools': tools, 'target_text': text,
                'target_token_ids': tokens.tolist(), 'tool_image_mode': tool_image_mode})
        turn = {'step': step, 'generation': text, 'seconds': time.time() - start,
                'input_tokens': int(inputs['input_ids'].shape[1]), 'output_tokens': len(tokens),
                'generation_attempts':attempts,'initial_output_budget':max_new_tokens}
        turn['input_fingerprints']={key:{'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for key,value in inputs.items()if isinstance(value,torch.Tensor)}
        if not stopped:
            # Never execute or append half a tool call as a valid assistant
            # history. It does not correspond to a teacher training action.
            turn.update(action='output_budget_exceeded',passed=False)
            turns.append(turn)
            (out/'trajectory.json').write_text(json.dumps({'model':model_name,'adapter':adapter,
                'instruction':instruction,'complete':False,'messages':messages,'turns':turns},indent=2))
            print('OUTPUT_BUDGET_EXCEEDED',json.dumps(attempts),flush=True)
            break
        try:
            name, arguments = parse_call(text)
            messages.append({'role': 'assistant', 'content': text})
            runtime_step=step+(restored_history['historical_actions_reexecuted']if restored_history else 0)
            feedback, passed = execute_action(name, arguments, out, runtime_step,
                system_profile in ('rgb','roi','rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'),
                system_profile in ('roi','roi_edit','clean_edit','integer_edit'),
                system_profile in ('rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'))
            messages.append(feedback)
            turn.update(action=name, arguments=arguments, passed=passed, feedback=feedback)
            complete = name == 'finish' and passed
        except ValueError as error:
            messages.append({'role': 'assistant', 'content': text})
            messages.append({'role': 'user', 'content': 'No valid action was received: ' + str(error) + ' Use the provided tool-call format.'})
            turn.update(action='parse_error', error=str(error))
        turns.append(turn)
        print(json.dumps({k: v for k, v in turn.items() if k not in ('generation', 'arguments', 'feedback')}), flush=True)
        (out / 'trajectory.json').write_text(json.dumps(
            {'model': model_name, 'adapter': adapter, 'instruction': instruction,
             'complete': complete, 'messages': messages, 'turns': turns}, indent=2))
        if complete:
            break
    summary = {'complete': complete, 'steps': len(turns),
               'actions': [t['action'] for t in turns],
               'model': model_name, 'adapter': adapter,
               'validation': json.loads(messages[-1]['content']) if complete else None,
               'system_profile': system_profile,
               'runtime_features':runtime_features,
               'tool_image_mode': tool_image_mode,
               'decoding': {'temperature': temperature, 'top_p': top_p, 'top_k': 20 if temperature else None,
                            'initial_max_new_tokens':max_new_tokens,'adaptive_max_new_tokens':6144,
                            'max_context':max_context},
               'system_sha256': hashlib.sha256(messages[0]['content'].encode()).hexdigest()}
    if restored_history:
        summary['supplied_prefix_diagnostic']=restored_history
        summary['not_initial_task_autonomous']=True
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if complete and resume_prefix is None:
        export_rollout_sft(messages, training_rows, out / 'sft_steps.jsonl',
                           model_name, adapter, tools, stop_id)
    print('AGENT_DEMO_COMPLETE', json.dumps(summary), flush=True)
    return summary, training_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--source', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=1800)
    parser.add_argument('--image-max-pixels', type=int, default=512 * 512)
    parser.add_argument('--system-profile', choices=['default', 'grounded', 'rgb', 'roi','rgb_checked','rgb_normalized','roi_edit','clean_edit','integer_edit'], default='default')
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--tool-image-mode', choices=['tool','user'], default='tool')
    parser.add_argument('--checkpoint-interface',action='store_true')
    args = parser.parse_args()
    model, processor = load(SimpleNamespace(model=args.model, adapter=args.adapter,
        mode='generate', image_max_pixels=args.image_max_pixels))
    rollout(model, processor, args.source, args.instruction, args.out,
        args.max_steps, args.max_new_tokens, temperature=args.temperature,
        model_name=args.model, adapter=args.adapter, system_profile=args.system_profile, top_p=args.top_p, tool_image_mode=args.tool_image_mode,use_checkpoint_interface=args.checkpoint_interface)


if __name__ == '__main__':
    main()
