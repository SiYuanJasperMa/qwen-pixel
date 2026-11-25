"""Execute the assistant-authored bus demonstration and record real feedback."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.agent_protocol import (assistant_call, execute_action, export_sft_steps,
                                  initial_messages)


def main():
    demonstration = ROOT / 'outputs/pixel_edit_demo'
    out = ROOT / 'outputs/agent_demo/teacher'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demonstration / 'source.png', out / 'source.png')
    instruction = 'Change the red bodywork of both buses to blue. Preserve the windows, wheels, road, and background.'
    inspect = '''rgb = np.asarray(Image.open('source.png').convert('RGB'))
print('shape:', rgb.shape, 'dtype:', rgb.dtype)
print('roof RGB:', rgb[229,319].tolist())
print('body RGB:', rgb[308,345].tolist())
print('window RGB:', rgb[265,348].tolist())
print('road RGB:', rgb[390,345].tolist())
Image.fromarray(rgb).crop((8,200,466,347)).save('bus_crop.png')'''
    source = (demonstration / 'edit_pixels.py').read_text()
    function = source[source.index('def edit_pixels('):source.index('\ndef main():')]
    edit = function + '''
rgb = np.asarray(Image.open('source.png').convert('RGB'))
edited, mask = edit_pixels(rgb)
np.save('edited_rgb.npy', edited)
Image.fromarray(edited).save('edited.png')
print('changed pixels:', int(mask.sum()))
print('saved edited.png and edited_rgb.npy')
'''
    actions = [('python', {'code': inspect}), ('view_image', {'path': 'bus_crop.png'}),
               ('python', {'code': edit}), ('view_image', {'path': 'edited.png'}),
               ('finish', {'image_path': 'edited.png', 'rgb_path': 'edited_rgb.npy'})]
    messages = initial_messages(out / 'source.png', instruction)
    for step, (name, arguments) in enumerate(actions):
        messages.append(assistant_call(name, arguments))
        feedback, passed = execute_action(name, arguments, out, step)
        messages.append(feedback)
        print(step, name, passed, flush=True)
        if not passed:
            raise RuntimeError(feedback)
    provenance = 'assistant_authored_GPT-6.1-Sol_demo_with_actual_replayed_tool_outputs; not an API teacher rollout'
    (out / 'trajectory.json').write_text(json.dumps(
        {'instruction': instruction, 'messages': messages, 'provenance': provenance}, indent=2))
    data = ROOT / 'data/agent_demo_v1'
    data.mkdir(parents=True, exist_ok=True)
    count = export_sft_steps(messages, data / 'train.jsonl', 'bus_demo', provenance)
    (data / 'manifest.json').write_text(json.dumps(
        {'trajectories': 1, 'assistant_actions': count, 'purpose': 'pipeline and overfit smoke only',
         'provenance': provenance}, indent=2))
    print('SFT_ACTIONS', count, flush=True)


if __name__ == '__main__':
    main()
