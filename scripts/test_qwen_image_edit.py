"""One-case smoke for Qwen-Image-2.1 instruction editing on MagicBrush."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig

ROOT = Path('/mnt/bn/embodied-lf3/masiyuan')
MODEL_BASE = ROOT / 'models_diffsynth'
CASE_ID = 'magic_dev_319096'
SRC = ROOT / 'qwen_image_edit_smoke/data/magicbrush_sample_v1/images/dev' / f'{CASE_ID}.png'
TGT = ROOT / 'qwen_image_edit_smoke/data/magicbrush_sample_v1/images/dev' / f'{CASE_ID}_target.png'
OUT = ROOT / 'qwen_pixel/outputs/qwen_image_edit_test'
OUT.mkdir(parents=True, exist_ok=True)

print('LOAD', flush=True)
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        ModelConfig(model_id='Qwen/Qwen-Image-2.1', origin_file_pattern='transformer/diffusion_pytorch_model*.safetensors', local_model_path=str(MODEL_BASE), skip_download=True),
        ModelConfig(model_id='Qwen/Qwen-Image-2.1', origin_file_pattern='text_encoder/model*.safetensors', local_model_path=str(MODEL_BASE), skip_download=True),
        ModelConfig(model_id='Qwen/Qwen-Image-2.1', origin_file_pattern='vae/diffusion_pytorch_model*.safetensors', local_model_path=str(MODEL_BASE), skip_download=True),
    ],
    processor_config=ModelConfig(model_id='Qwen/Qwen-Image-2.1', origin_file_pattern='processor/', local_model_path=str(MODEL_BASE), skip_download=True),
    vram_limit=None,
)
print('LOADED', round(time.time() - t0, 1), 'sec', flush=True)
instruction = 'Make the piece of paper hanging on the wall a mirror'
source = Image.open(SRC).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
t1 = time.time()
edited = pipe(prompt=instruction, edit_image=source, height=512, width=512,
              num_inference_steps=20, seed=0, cfg_scale=1.0)
print('GENERATED', round(time.time() - t1, 1), 'sec', flush=True)
edited.save(OUT / f'{CASE_ID}_edited.png')
source.save(OUT / f'{CASE_ID}_source.png')
Image.open(TGT).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS).save(OUT / f'{CASE_ID}_target.png')
panel = Image.new('RGB', (1536, 512), 'white')
panel.paste(source, (0, 0))
panel.paste(Image.open(TGT).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
panel.paste(edited, (1024, 0))
panel.save(OUT / f'{CASE_ID}_compare.png')
print(json.dumps({'case': CASE_ID, 'instruction': instruction, 'output': str(OUT / f'{CASE_ID}_compare.png')}), flush=True)
