"""Freeze a small evaluation-only real-photo editing set from COCO val2017.

Reads only the validation annotation entry through HTTP ranges; does not
download the full annotation archive or create any teacher training examples.
"""
import concurrent.futures
import hashlib
import io
import json
import math
import re
from pathlib import Path
from urllib.request import Request,urlopen
import zipfile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'


class RangeReader(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.position=0;self.cache=None
        with urlopen(Request(url,method='HEAD'),timeout=30)as response:
            self.length=int(response.headers['Content-Length'])
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if self.position<0:raise ValueError('Negative archive position.')
        return self.position
    def read(self,size=-1):
        size=self.length-self.position if size<0 else min(size,self.length-self.position)
        if size<=0:return b''
        if size>64*1024*1024:raise ValueError('Refuse to fetch an unexpectedly large archive range.')
        start=self.position;end=start+size
        if self.cache and self.cache[0]<=start and end<=self.cache[1]:
            data=self.cache[2][start-self.cache[0]:end-self.cache[0]]
        else:
            fetch_start=(start//65536)*65536 if size<=65536 else start
            fetch_end=min(self.length,max(end,fetch_start+65536))if size<=65536 else end
            request=Request(self.url,headers={'Range':f'bytes={fetch_start}-{fetch_end-1}'})
            with urlopen(request,timeout=60)as response:
                if response.status!=206:raise ValueError('Annotation source did not honor range reads.')
                raw=response.read()
            if len(raw)!=fetch_end-fetch_start:raise ValueError('Incomplete annotation range.')
            self.cache=(fetch_start,fetch_end,raw)
            data=raw[start-fetch_start:end-fetch_start]
        self.position=end
        return data


def download(task,out):
    image_path=out/'images'/task['file_name']
    if not image_path.exists():
        with urlopen(task['source_url'],timeout=30)as response:raw=response.read()
        with Image.open(io.BytesIO(raw))as image:image.verify()
        image_path.write_bytes(raw)
    return {**task,'image':str(image_path),'source_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()}


def main():
    out=ROOT/'data/independent_coco_edit_eval_v1';(out/'images').mkdir(parents=True,exist_ok=True)
    annotation=out/'instances_val2017.json'
    if not annotation.exists():
        with zipfile.ZipFile(RangeReader(ARCHIVE))as archive:
            info=archive.getinfo('annotations/instances_val2017.json')
            print('VALIDATION_ANNOTATION_RANGE',info.compress_size,flush=True)
            annotation.write_bytes(archive.read(info.filename))
    data=json.loads(annotation.read_text());images={i['id']:i for i in data['images']}
    names={c['id']:c['name']for c in data['categories']}
    excluded=set()
    for line in(ROOT/'data/edit_tasks_coco_v1/tasks.jsonl').open():
        task=json.loads(line);scene=task.get('meta',{}).get('scene_id')
        if scene is not None:excluded.add(int(scene))
    candidates=[];counts={}
    for a in data['annotations']:
        if a.get('iscrowd')or a['image_id']in excluded:continue
        image=images[a['image_id']];x,y,w,h=a['bbox']
        fraction=w*h/(image['width']*image['height'])
        if not(.04<=fraction<=.65)or a['area']/(w*h)<.4:continue
        counts.setdefault((a['image_id'],a['category_id']),[]).append(a)
    for key,annotations in counts.items():
        annotations.sort(key=lambda a:a['area'],reverse=True)
        if len(annotations)>2:continue
        if len(annotations)==2 and annotations[0]['area']<1.5*annotations[1]['area']:continue
        a=annotations[0]
        order=hashlib.sha256(f'20261118:{a["image_id"]}:{a["id"]}'.encode()).hexdigest()
        candidates.append((order,a,len(annotations)))
    candidates.sort(key=lambda item:item[0])
    tint_categories={'bicycle','car','motorcycle','bus','train','truck','boat','backpack',
        'umbrella','handbag','suitcase','chair','couch','bottle','vase'}
    selected=[];used=set();category_counts={}
    for i in range(20):
        operation=('blur','brighten','recolor')[i%3]
        chosen=None
        for _,a,count in candidates:
            name=names[a['category_id']]
            if a['image_id']in used or category_counts.get(name,0)>=3:continue
            if operation=='recolor'and name not in tint_categories:continue
            chosen=(a,count,name);break
        if chosen is None:raise ValueError('Insufficient independent evaluation candidates.')
        a,count,name=chosen;used.add(a['image_id']);category_counts[name]=category_counts.get(name,0)+1
        image=images[a['image_id']]
        if not re.fullmatch(r'\d{12}\.jpg',image['file_name']):raise ValueError('Unexpected COCO filename.')
        target=('the largest '+name)if count>1 else('the '+name)
        instruction=(f'Blur {target}, keeping the rest of the image unchanged.'if operation=='blur'else
            f'Make {target} moderately brighter while preserving its details and keeping the rest unchanged.'if operation=='brighten'else
            f'Apply a subtle red tint to {target}, retaining its texture and shape and keeping the rest unchanged.')
        x,y,w,h=a['bbox']
        selected.append({'id':f'coco_val2017_edit_{i:02d}','file_name':image['file_name'],
            'image_id':a['image_id'],'original_scene_id':str(a['image_id']),'split':'val',
            'instruction':instruction,'operation':operation,'category':name,
            'source_url':'https://s3.amazonaws.com/images.cocodataset.org/val2017/'+image['file_name'],
            'evaluation_only':True,'annotation_id':a['id'],
            'target_bbox_pixels':[math.floor(x),math.floor(y),math.ceil(x+w),math.ceil(y+h)]})
    tasks=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
        futures=[pool.submit(download,t,out)for t in selected]
        for future in concurrent.futures.as_completed(futures):
            task=future.result();tasks.append(task);print('EVAL_IMAGE',task['id'],task['category'],flush=True)
    tasks.sort(key=lambda t:t['id'])
    text=''.join(json.dumps(task)+'\n'for task in tasks)
    (out/'tasks.jsonl').write_text(text)
    manifest={'images':len(tasks),'official_image_split':'COCO val2017','training_image_id_overlap':0,
        'category_counts':category_counts,'tasks_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
        'evaluation_only':True,'seed':20261118,
        'note':'Custom local editing instructions on official validation photos; not an official COCO editing benchmark. Frozen before evaluation; do not use in SFT or teacher training.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('INDEPENDENT_EDIT_EVAL_FROZEN',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
