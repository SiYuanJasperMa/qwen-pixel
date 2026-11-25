"""Keep the teacher reservoir supplied from public TRAIN editing triplets."""
import argparse
import concurrent.futures as cf
import hashlib
import io
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from continuous_editing_teachers import rows, atomic_json
from pixel.data_lineage import split_of_group
from pixel.data_language import english_task
from pixel.source_policy import disabled_sources

SOURCES = [
    ('anyedit', 'Bin1117/AnyEdit', 'data/train-', 'image_file', 'edited_file', 'edit_instruction'),
    ('maskedit_scene', 'ReyChiaro/MaskEdit-10k', 'data/scene/train-', 'source', 'target', 'edit_instruction'),
    ('maskedit_en', 'ReyChiaro/MaskEdit-10k', 'data/infographics_en/train-', 'source', 'target', 'edit_instruction'),
    ('sciforma', 'microsoft/SciFormaData-700K', 'data/editing/train-', 'source_image', 'target_image', 'caption'),
    ('ultraedit', 'BleachNick/UltraEdit_Region_Based_100k', 'data/RegionBase-', 'source_image', 'edited_image', 'edit_prompt'),
    ('magicbrush', 'osunlp/MagicBrush', 'data/train-', 'source_img', 'target_img', 'instruction'),
]


def rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)


def digest(image): return hashlib.sha256(image.tobytes()).hexdigest()


def save_png(image,path):
    if path.is_file():return
    temp=path.with_name(path.stem+'.tmp.png')
    try:
        image.save(temp);temp.replace(path)
    finally:
        if temp.is_file():temp.unlink()


def raw_image(value, parent):
    if isinstance(value, dict):
        if value.get('bytes'): return Image.open(io.BytesIO(value['bytes'])).convert('RGB')
        value = value.get('path')
    if isinstance(value, bytes): return Image.open(io.BytesIO(value)).convert('RGB')
    if isinstance(value, str) and (parent / value).is_file(): return Image.open(parent / value).convert('RGB')
    raise ValueError('Missing embedded image bytes')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--backlog',type=int,default=10000)
    parser.add_argument('--chunk-rows',type=int,default=256)
    parser.add_argument('--image-budget-gib',type=float,default=8.)
    args=parser.parse_args()
    if args.image_budget_gib<=0:raise ValueError('Image budget must be positive')
    out=ROOT/'data/editing_growth_v1'; out.mkdir(exist_ok=True)
    lock=(out/'ingestion.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(out/'ingestion_pid.json',{'pid':os.getpid(),'started':time.time()})
    images=out/'images'; images.mkdir(exist_ok=True)
    state_path=out/'ingestion_state.json'; state=json.loads(state_path.read_text())if state_path.is_file()else{}
    state.setdefault('completed_files',[]); state.setdefault('counts',{}); state.setdefault('errors',{})
    state.setdefault('file_cursors',{})
    official_ids={t['id'].split('_')[2]for t in rows(ROOT/'data/magicbrush_official_dev_v1/tasks.jsonl')}
    known={}; seen=set()
    task_rows=[r for split in ('train','val')for r in rows(ROOT/f'data/star_clean_v2/tasks_{split}.jsonl')]
    task_rows += rows(out/'tasks_train.jsonl') + rows(out/'tasks_val.jsonl')
    seen.update(r['id']for r in task_rows)
    index_path=out/'known_rgb_groups.json'
    if index_path.is_file():
        known=json.loads(index_path.read_text())
        # Completed manifest lines survive a crash even if the index snapshot lags.
        for task in task_rows:
            for field in('source_rgb_sha256','target_rgb_sha256'):
                if task.get(field):
                    known.setdefault(task[field],{'group':task['source_group_id'],'split':task['split']})
        atomic_json(index_path,known)
    else:
        paths={}
        for t in task_rows:
            for key in ('image','target_image'):
                if t.get(key):paths[t[key]]={'group':t['source_group_id'],'split':t['split']}
        for t in rows(ROOT/'data/magicbrush_official_dev_v1/tasks.jsonl'):
            for key in ('image','target_image'):
                path=Path(t[key]);path=path if path.is_absolute()else ROOT/path
                paths[str(path)]={'group':'official_magicbrush_dev','split':'excluded'}
        atomic_json(out/'ingestion_status.json',{'phase':'indexing_existing_rgb','images':len(paths)})
        def hash_path(item):
            path,meta=item
            return digest(rgb(path)),meta
        with cf.ThreadPoolExecutor(max_workers=8)as pool:
            for i,(h,meta)in enumerate(pool.map(hash_path,paths.items())):
                prior=known.get(h)
                if prior and prior['split']!=meta['split']:
                    # Blocking shared content is safer than moving frozen validation groups.
                    known[h]={'group':'blocked_split_bridge','split':'excluded'}
                elif not prior or meta['split']=='excluded':known[h]=meta
                if i%2000==0:
                    print('RGB_INDEX',i,len(paths),flush=True)
                    atomic_json(out/'rgb_index_progress.json',{'images_completed':i,'images_total':len(paths)})
        atomic_json(index_path,known)

    def save_task(tid,dataset,instruction,source,target,origin):
        if dataset in disabled_sources():return
        if tid in seen or not isinstance(instruction,str)or not instruction.strip():return
        if not english_task({'dataset':dataset,'instruction':instruction}):
            state['excluded_non_english']=state.get('excluded_non_english',0)+1;seen.add(tid);return
        source_size=source.size; target_size=target.size
        source=source.resize((512,512),Image.Resampling.LANCZOS)
        target=target.resize((512,512),Image.Resampling.LANCZOS)
        sh,th=digest(source),digest(target)
        if sh==th:
            state['excluded_noop_pairs']=state.get('excluded_noop_pairs',0)+1;seen.add(tid);return
        matches=[known[h]for h in (sh,th)if h in known]
        if any(m['split']in('val','excluded')for m in matches):
            state['excluded_validation_content']=state.get('excluded_validation_content',0)+1;seen.add(tid);return
        group=matches[0]['group']if matches else 'normalized_rgb_sha256/'+sh
        split=matches[0]['split']if matches else split_of_group(group)
        meta={'group':group,'split':split}
        known[sh]=meta;known[th]=meta
        sp=images/(sh+'.png');tp=images/(th+'.png')
        save_png(source,sp);save_png(target,tp)
        task={'id':tid,'dataset':dataset,'instruction':instruction,'image':str(sp),'source_image':str(sp),
              'target_image':str(tp),'source_group_id':group,'split':split,'original_split':'train',
              'origin':origin,'normalization':'RGB resize 512x512','original_source_size':source_size,
              'original_target_size':target_size,'source_rgb_sha256':sh,'target_rgb_sha256':th}
        with(out/f'tasks_{split}.jsonl').open('a',buffering=1)as handle:handle.write(json.dumps(task,ensure_ascii=False)+'\n')
        seen.add(tid);state['counts'][dataset]=state['counts'].get(dataset,0)+1

    # The old AnyEdit pilot was from visual_depth, which needs another input.
    # Do not reuse those records as source-plus-instruction training tasks.
    atomic_json(index_path,known);atomic_json(state_path,state)
    print('CACHED_PAIRS_READY',state['counts'],flush=True)
    metadata={}; retries={}; fs=HfFileSystem(endpoint='https://hf-mirror.com',token=False)
    while True:
        progressed=False
        for dataset,repo,prefix,sc,tc,ic in SOURCES:
            if dataset in disabled_sources():continue
            if retries.get(dataset,0)>time.time():continue
            train_count=len(rows(out/'tasks_train.jsonl'))
            prod=json.loads((out/'status.json').read_text())if(out/'status.json').is_file()else{}
            completed=prod.get('fully_processed_tasks',sum(1 for r in rows(out/'judged.jsonl')if r.get('pipeline_qualified')))
            if train_count+3302-completed>=args.backlog:
                atomic_json(out/'ingestion_status.json',{'phase':'reservoir_full','new_train_tasks':train_count,'counts':state['counts']})
                time.sleep(60);break
            stored=sum(p.stat().st_size for p in images.glob('*.png'))
            if stored>args.image_budget_gib*1024**3:
                atomic_json(out/'ingestion_status.json',{'phase':'image_storage_budget','bytes':stored,'budget_gib':args.image_budget_gib,'counts':state['counts']})
                time.sleep(60);break
            try:
                if repo not in metadata:
                    response=requests.get('https://hf-mirror.com/api/datasets/'+repo,timeout=30);response.raise_for_status()
                    metadata[repo]=response.json()
                files=sorted(s['rfilename']for s in metadata[repo].get('siblings',[])if s['rfilename'].startswith(prefix)and s['rfilename'].endswith('.parquet'))
                if dataset=='anyedit':
                    files.sort(key=lambda f:(int(re.search(r'-(\d+)-of-',f).group(1))<200,f))
                remaining=[f for f in files if repo+':'+f not in state['completed_files']]
                if not remaining:continue
                filename=remaining[0]
                atomic_json(out/'ingestion_status.json',{'phase':'downloading_training_shard','dataset':dataset,'file':filename,'counts':state['counts']})
                # Range-read the parquet rather than keeping gigabyte raw shards.
                remote=fs.open('datasets/'+repo+'/'+filename,'rb')
                parquet=pq.ParquetFile(remote)
                available=set(parquet.schema_arrow.names)
                if dataset=='anyedit':
                    edit_type=next(parquet.iter_batches(batch_size=1,columns=['edit_type'])).to_pylist()[0]['edit_type']
                    if edit_type.startswith('visual_'):
                        state['completed_files'].append(repo+':'+filename)
                        state['skipped_reference_shards']=state.get('skipped_reference_shards',0)+1
                        remote.close();atomic_json(state_path,state);continue
                # InstructPix2Pix distributions use either original_image or input_image.
                src=sc if sc in available else('input_image'if'input_image'in available else sc)
                tgt=tc if tc in available else('output_image'if'output_image'in available else tc)
                ins=ic if ic in available else('instruction'if'instruction'in available else ic)
                if not{src,tgt,ins}<=available:raise ValueError('Unsupported editing-triplet schema: '+','.join(sorted(available)))
                shard=int(re.search(r'-(\d+)-of-',filename).group(1))
                offset=0; cursor=state['file_cursors'].get(repo+':'+filename,0)
                columns=list(dict.fromkeys([src,tgt,ins]+[c for c in('idx','img_id','turn_index','edit_type')if c in available]))
                for batch in parquet.iter_batches(batch_size=16,columns=columns):
                    for record in batch.to_pylist():
                        if dataset=='magicbrush':tid=f"magicbrush_train_{record['img_id']}_{record['turn_index']}"
                        elif dataset=='sciforma':tid=f'sciforma_s{shard}_{offset}'
                        elif dataset=='ultraedit':tid=f"ultraedit_{record['idx']}"
                        elif dataset=='anyedit':tid=f'anyedit_{offset}'if shard==0 else f'anyedit_s{shard}_{offset}'
                        else:tid=f'{dataset}_s{shard}_{offset}'
                        offset+=1
                        if offset<=cursor:continue
                        if tid in seen:continue
                        if dataset=='magicbrush'and str(record['img_id'])in official_ids:
                            seen.add(tid);state['excluded_official_image_id']=state.get('excluded_official_image_id',0)+1;continue
                        if dataset=='anyedit'and record.get('edit_type','').startswith('visual_'):
                            seen.add(tid);state['excluded_reference_tasks']=state.get('excluded_reference_tasks',0)+1;continue
                        if not english_task({'dataset':dataset,'instruction':record[ins]}):
                            seen.add(tid);state['excluded_non_english']=state.get('excluded_non_english',0)+1;continue
                        try:save_task(tid,dataset,record[ins],raw_image(record[src],out),raw_image(record[tgt],out),
                                      {'repo':repo,'file':filename,'row':offset-1,'license':metadata[repo].get('cardData',{}).get('license')})
                        except ValueError:state['invalid_image_rows']=state.get('invalid_image_rows',0)+1
                        except OSError as exc:
                            if exc.errno in(28,122):
                                state['file_cursors'][repo+':'+filename]=offset-1
                                atomic_json(state_path,state);raise
                            state['invalid_image_rows']=state.get('invalid_image_rows',0)+1
                    state['file_cursors'][repo+':'+filename]=offset
                    atomic_json(state_path,state)
                    if offset>=cursor+args.chunk_rows:break
                if offset>=parquet.metadata.num_rows:state['completed_files'].append(repo+':'+filename)
                remote.close()
                atomic_json(index_path,known);atomic_json(state_path,state)
                print('SHARD_ADDED',dataset,filename,state['counts'].get(dataset,0),flush=True);progressed=True
            except Exception as exc:
                state['errors'][dataset]={'type':type(exc).__name__,'retry_after':time.time()+900}
                retries[dataset]=time.time()+900
                atomic_json(state_path,state)
                print('DATASET_RETRY',dataset,type(exc).__name__,flush=True)
        if not progressed:time.sleep(60)


if __name__=='__main__':main()
