"""Small source-RGB mask representation pilot on existing approved train photos.

Annotations choose training-output seed coordinates, never a runtime mask input.
This produces candidate supervision evidence, not accepted SFT or teacher traces.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from pixel.image_observations import load_observation
from pixel.agent_protocol import execute_action, validate_outputs
from pixel.clean_edit_protocol import initial, program
from pixel.roi_augmentation import padded_box, normalized_box
from build_semantic_repair_sft import raster


def seeds(mask, count, radius):
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    points = np.argwhere(distance > radius+1)
    if len(points) < count:
        raise ValueError('Insufficient safe seed interiors.')
    boundary = distance[points[:,0], points[:,1]]
    selected = [int(np.argmax(boundary))]
    minimum = np.full(len(points), np.inf)
    for _ in range(count-1):
        last = points[selected[-1]]
        minimum = np.minimum(minimum, np.sum((points-last)**2, axis=1))
        selected.append(int(np.argmax(minimum*np.sqrt(boundary))))
    height, width = mask.shape
    return [[round(float(points[i,1]/width),4), round(float(points[i,0]/height),4)] for i in selected]


def mask_program(coarse, foreground, background, operation, color):
    code = program(coarse, operation, color)
    marker = 'soft_mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=0.7))'
    if code.count(marker) != 1:
        raise ValueError('Unexpected native program format.')
    refine = '''import cv2
cv2.setNumThreads(1)
cv2.setRNGSeed(0)
source_roi_uint8 = np.ascontiguousarray(pixels[y0:y1,x0:x1])
coarse_roi_np = np.asarray(mask_pil)[y0:y1,x0:x1]
gc_labels_np = np.where(coarse_roi_np>0,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
foreground_points_roi = '''+repr(foreground)+'''
background_points_roi = '''+repr(background)+'''
seed_radius = max(1,min(4,int(round(min(region_width,region_height)*0.01))))
for points_roi,label in [(foreground_points_roi,cv2.GC_FGD),(background_points_roi,cv2.GC_BGD)]:
    for u,v in points_roi:
        point_x = min(region_width-1,max(0,int(round(u*region_width))))
        point_y = min(region_height-1,max(0,int(round(v*region_height))))
        cv2.circle(gc_labels_np,(point_x,point_y),seed_radius,label,-1)
background_model_np = np.zeros((1,65),dtype=np.float64)
foreground_model_np = np.zeros((1,65),dtype=np.float64)
cv2.grabCut(source_roi_uint8,gc_labels_np,None,background_model_np,foreground_model_np,5,cv2.GC_INIT_WITH_MASK)
full_mask_np = np.zeros((height,width),dtype=np.uint8)
full_mask_np[y0:y1,x0:x1] = np.where((gc_labels_np==cv2.GC_FGD)|(gc_labels_np==cv2.GC_PR_FGD),255,0).astype(np.uint8)
np.save('computed_mask.npy',full_mask_np,allow_pickle=False)
mask_pil = Image.fromarray(full_mask_np)
'''
    return code.replace(marker, refine+marker)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--photos', type=int, default=8)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Choose a fresh pilot output.')
    args.out.mkdir(parents=True)
    rows = [json.loads(l) for l in (ROOT/'data/agent_sft_en_27b_37photos_20261125/train.jsonl').read_text().splitlines()]
    tasks = {r['id']:r for r in map(json.loads,(ROOT/'data/sft_clean_mask_v3/tasks.jsonl').read_text().splitlines())}
    programs = {r['meta']['trajectory_id']:r for r in rows
                if r['target_message']['tool_calls'][0]['function']['name']=='python'}
    selected, seen = [], set()
    for identity, row in programs.items():
        task = tasks[identity]
        if task['original_scene_id'] in seen:
            continue
        seen.add(task['original_scene_id']);selected.append((identity,row,task))
        if len(selected) >= args.photos:
            break
    results = []
    for identity, row, task in selected:
        folder = args.out/identity;folder.mkdir()
        case = {'id':identity,'operation':task['operation'],'not_added_to_sft':True}
        try:
            source_block = next(b for m in row['messages'] if m['role']=='user'
                                for b in m['content'] if b.get('type')=='image')
            image = load_observation(source_block)
            code = row['target_message']['tool_calls'][0]['function']['arguments']['code']
            original_box = next(json.loads(b['text'])['source_bbox_pixels']
                for m in reversed(row['messages']) if m['role']=='tool' and isinstance(m['content'],list)
                for b in m['content'] if b.get('type')=='text')
            tree = ast.parse(code)
            polygons = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='polygons_roi' for t in n.targets))
            reference = raster(polygons,original_box,image.size)
            Image.fromarray(reference.astype(np.uint8)*255).save(folder/'approved_polygon_reference.png')
            runtime = folder/'actor';runtime.mkdir();image.save(runtime/'source.png')
            initial(runtime/'source.png',task['instruction'])
            requested_box = padded_box(original_box,image.size,.12)
            _, passed = execute_action('view_image',{'path':'source.png','bbox_2d':normalized_box(requested_box,image.size)},runtime,0,True,True,True)
            if not passed:
                raise ValueError('Source crop failed.')
            x0,y0,x1,y1 = json.loads((runtime/'region.json').read_text())['source_bbox_pixels']
            crop_mask = reference[y0:y1,x0:x1]
            radius = max(1,min(4,int(round(min(x1-x0,y1-y0)*.01))))
            foreground = seeds(crop_mask,8,radius)
            background = seeds(~crop_mask,8,radius)
            ys,xs = np.nonzero(reference)
            bx0,by0,bx1,by1 = int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
            coarse = [[[(bx0-x0)/(x1-x0),(by0-y0)/(y1-y0)],[(bx1-x0)/(x1-x0),(by0-y0)/(y1-y0)],
                       [(bx1-x0)/(x1-x0),(by1-y0)/(y1-y0)],[(bx0-x0)/(x1-x0),(by1-y0)/(y1-y0)]]]
            candidate_code = mask_program(coarse,foreground,background,task['operation'],task.get('tint_color') or 'red')
            (folder/'program.py').write_text(candidate_code)
            _, passed = execute_action('python',{'code':candidate_code},runtime,1,True,True,True)
            if not passed:
                raise ValueError('Source-only RGB program failed.')
            receipt = validate_outputs(runtime)
            output = np.load(runtime/'edited_rgb.npy',allow_pickle=False).copy()
            prediction = np.load(runtime/'computed_mask.npy',allow_pickle=False)>0
            _, second = execute_action('python',{'code':candidate_code},runtime,2,True,True,True)
            exact = second and np.array_equal(output,np.load(runtime/'edited_rgb.npy',allow_pickle=False))
            allowed = np.asarray(Image.fromarray(reference.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
            intersection = np.count_nonzero(prediction&reference)
            union = np.count_nonzero(prediction|reference)
            recall = intersection/max(1,np.count_nonzero(reference))
            outside = np.count_nonzero(prediction&~allowed)/max(1,np.count_nonzero(prediction))
            case.update(execution_passed=True,repeat_rgb_exact=bool(exact),mask_iou=intersection/max(1,union),
                reference_mask_recall=recall,outside_mask_fraction_boundary2=outside,
                local_quality_candidate=bool(exact and recall>=.95 and intersection/max(1,union)>=.9 and outside<=.03),
                rgb_changed_pixels=receipt['changed_pixels'],foreground_seed_points=8,background_seed_points=8)
        except Exception as error:
            case.update(execution_passed=False,error=type(error).__name__+': '+str(error))
        results.append(case)
        (args.out/'results.json').write_text(json.dumps(results,indent=2))
        print('PIXEL_MASK_PILOT',json.dumps(case),flush=True)
    report = {'cases':results,'photos':len(results),'executed':sum(r.get('execution_passed',False) for r in results),
        'local_quality_candidates':sum(r.get('local_quality_candidate',False) for r in results),
        'training_data_ready':False,'new_teacher_calls':0,
        'scope':'annotation-derived source-only code feasibility; approved polygon masks are references, not new official GT evaluation',
        'next_gate':'visual/semantic review and rebuilt actual observation histories before any SFT'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='cases'}),flush=True)


if __name__ == '__main__':
    main()
