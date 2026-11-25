"""Save real source/result crops and measured changes for manual label review."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(1000,len(args.ids)*330),(245,245,245));draw=ImageDraw.Draw(canvas);results=[]
    for index,identity in enumerate(args.ids):
        folder=args.data/'executions'/identity
        region=json.loads((folder/'region.json').read_text())['source_bbox_pixels']
        with Image.open(folder/'source.png')as image:source=image.convert('RGB').crop(tuple(region))
        with Image.open(folder/'edited.png')as image:edited=image.convert('RGB').crop(tuple(region))
        a=np.asarray(source).astype(float);b=np.asarray(edited).astype(float);difference=np.abs(b-a)
        changed=np.any(a!=b,axis=2)
        stats={'id':identity,'changed_crop_fraction':float(changed.mean()),'crop_mae':float(difference.mean()),
            'mean_changed_rgb_delta':float(difference[changed].mean())if changed.any()else 0.,
            'mean_brightness_increase':float((b-a).mean())}
        results.append(stats)
        amplified=Image.fromarray(np.clip(difference*6,0,255).astype(np.uint8))
        draw.text((10,index*330+4),identity+'  MAE='+str(round(stats['crop_mae'],2)),fill='black')
        for column,image in enumerate((source,edited,amplified)):
            image.thumbnail((310,285));canvas.paste(image,(column*330+10,index*330+35))
    canvas.save(args.out/'montage.png');(args.out/'stats.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
