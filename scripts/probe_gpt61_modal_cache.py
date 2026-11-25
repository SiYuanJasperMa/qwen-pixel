"""Measure cold/repeated request usage; no edits are executed or repaired."""
import concurrent.futures as cf
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_rgb_text_teacher import call_teacher
from pixel.persistent_io import atomic_text


def main():
    folder=ROOT/'outputs/gpt61_three_settings_v1'
    out=folder/'cache_probes';out.mkdir(exist_ok=True)
    key=(ROOT.parent/'traex_api_proxy/proxy_logs_rgb_gpt61/local.key').read_text().strip()
    tid='magicbrush_dev_7730_2'
    def probe(mode):
        body=json.loads((folder/mode/tid/'request.json').read_text())
        body['prompt_cache_key']=f'gpt61_cost_probe_v1:{mode}:{tid}'
        results=[]
        for phase in ['cold','warm']:
            record={'mode':mode,'phase':phase,'task_id':tid}
            try:
                _,response=call_teacher(body,key)
                record.update(response_id=response['id'],usage=response.get('usage'),status=response['status'])
                atomic_text(out/f'{mode}_{phase}_response.json',json.dumps(response,ensure_ascii=False))
            except Exception as exc:record.update(status='api_error',error=type(exc).__name__+': '+str(exc)[:200])
            results.append(record)
            atomic_text(out/f'{mode}.json',json.dumps(results,indent=2))
            print('CACHE_PROBE',mode,phase,record.get('usage'),record.get('error'),flush=True)
        return results
    with cf.ThreadPoolExecutor(max_workers=3)as pool:
        outputs=list(pool.map(probe,['image_only','image_rgb','rgb_text']))
    atomic_text(out/'results.json',json.dumps([r for rr in outputs for r in rr],indent=2))


if __name__=='__main__':main()
