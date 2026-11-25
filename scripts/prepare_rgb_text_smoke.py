"""Prepare real editing tasks with exact RGB text and equal-resolution VL inputs."""
import json
import re
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pixel.rgb_text import encode_rgb, decode_rgb
import numpy as np


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def absolute(path):
    p = Path(path); return p if p.is_absolute() else ROOT / p


def main():
    out = ROOT / 'data/rgb_text_smoke_v1'; out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ROOT.parent / 'robojudge_v4q/models/Qwen3.8-27B')
    official = read(ROOT / 'data/magicbrush_official_dev_v1/tasks.jsonl')
    colors = [r for r in official if re.search(r'\b(blue|red|green|yellow|white|black|color|colour)\b',r['instruction'],re.I)
              and re.search(r'\b(change|make|turn|let)\b',r['instruction'],re.I)]
    chosen = [(r,128) for r in colors[:2]]
    scientific = [r for r in read(ROOT/'data/star_clean_v2/tasks_val.jsonl') if r['dataset']=='sciforma']
    chosen += [(scientific[0],512)] if scientific else []
    tasks = []
    for r, side in chosen:
        sid = r['id']; source = out/f'{sid}_source.png'; target = out/f'{sid}_target.png'
        for src, dest in [(r['image'],source),(r['target_image'],target)]:
            with Image.open(absolute(src)) as im:
                im.convert('RGB').resize((side,side),Image.Resampling.LANCZOS).save(dest)
        image = Image.open(source).convert('RGB')
        text = encode_rgb(image,'hex_rle')
        pixels = len(tok.encode(text))
        # Keep 512px diagrams when lossless RLE fits. No quantization.
        if pixels > 120000 and side>128:
            side=128
            for path in (source,target):
                with Image.open(path) as im: resized=im.resize((side,side),Image.Resampling.LANCZOS)
                resized.save(path)
            image=Image.open(source).convert('RGB');text=encode_rgb(image,'hex_rle');pixels=len(tok.encode(text))
        assert np.array_equal(np.asarray(image),np.asarray(decode_rgb(text)))
        rgb = out/f'{sid}.rgb.txt';rgb.write_text(text)
        tasks.append({'id':sid,'dataset':r.get('dataset','magicbrush'),'instruction':r['instruction'],
                      'image':str(source),'target_image':str(target),'rgb_text':str(rgb),
                      'width':side,'height':side,'rgb_tokens':pixels,'evaluation_only':True})
        print('RGB_TASK',json.dumps(tasks[-1]),flush=True)
    (out/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))


if __name__=='__main__':main()
