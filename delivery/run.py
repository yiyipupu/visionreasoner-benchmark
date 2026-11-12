import argparse, json, os, sys, traceback
from pathlib import Path
ROOT = Path(os.environ.get('VISIONREASONER_ROOT', '/data/yukun/VisionReasoner'))
for name in ('env','cache','checkpoints','datasets','outputs','weights'):
    (ROOT/name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(ROOT/'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'cache/torch'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vision_reasoner'))

def records(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids=set()
    for r in rows:
        assert r['id'] not in ids, 'duplicate id'
        ids.add(r['id'])
        assert r['task'] in ('counting','segmentation','detection','vqa')
        assert isinstance(r['query'],str) and r['query']
        r['image']=str((Path(path).resolve().parent/r['image']).resolve())
        from PIL import Image
        with Image.open(r['image']) as im: size=im.size
        if 'mask' in r:
            r['mask']=str((Path(path).resolve().parent/r['mask']).resolve())
            import numpy as np
            with Image.open(r['mask']) as m:
                a=np.asarray(m); assert m.size==size and a.ndim==2
                assert set(np.unique(a)).issubset({0,1,255})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('--validate',action='store_true'); p.add_argument('--evaluate',action='store_true'); p.add_argument('--output',default=str(ROOT/'outputs/results'))
    args=p.parse_args(); rows=records(args.manifest); out=Path(args.output)
    if args.validate: print(f'Validated {len(rows)} samples'); return
    out.mkdir(parents=True,exist_ok=True)
    if args.evaluate:
        import numpy as np
        from PIL import Image
        results=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
        lookup={r['id']:r for r in rows}; ious=[]; errors=[]; failures=0
        for result in results:
            if result['status']!='ok': failures+=1; continue
            r=lookup[result['id']]
            if r['task']=='segmentation' and 'mask' in r:
                gt=np.array(Image.open(r['mask'])); pred=np.array(Image.open(out/result['mask']))>0
                valid=gt!=255; target=gt==1
                intersection=(pred & target & valid).sum(); union=((pred|target)&valid).sum()
                if valid.any(): ious.append(float(intersection/union) if union else 1.)
            if r['task']=='counting' and 'count' in r: errors.append(abs(result['count']-r['count']))
        print(json.dumps({'samples':len(rows),'recorded':len(results),'failed':failures,'missing':len(set(lookup)-{r['id'] for r in results}),'segmentation_scored':len(ious),'mean_IoU':float(np.mean(ious)) if ious else None,'count_scored':len(errors),'count_MAE':float(np.mean(errors)) if errors else None},indent=2)); return
    import torch
    from models.vision_reasoner_model import VisionReasonerModel
    from models.task_router import TaskRouter
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from PIL import Image
    import numpy as np
    # Bypass upstream eager initialization; reuse its unchanged task methods/preprocessing.
    model=VisionReasonerModel.__new__(VisionReasonerModel)
    model.resize_size=840; model.use_hybrid_mode=False
    model.reasoning_model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ROOT/'weights/VisionReasoner-7B'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    model.processor=AutoProcessor.from_pretrained(str(ROOT/'weights/VisionReasoner-7B'),padding_side='left',use_fast=False)
    model.segmentation_model=SAM2ImagePredictor(build_sam2('configs/sam2/sam2_hiera_l.yaml',str(ROOT/'weights/sam2-hiera-large/sam2_hiera_large.pt'),device='cuda:0')) if any(r['task']=='segmentation' for r in rows) else None
    model.DETECTION_TEMPLATE='Please find "{Question}" with bboxs and points.Compare the difference between object(s) and find the most closely matched object(s).Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.Output the bbox(es) and point(s) inside the interested object(s) in JSON format.i.e., <think> thinking process here </think><answer>{Answer}</answer>'
    model.QA_TEMPLATE='{Question}'
    methods={'segmentation':model.segment_objects,'detection':model.detect_objects,'counting':model.count_objects,'vqa':model.answer_question}
    with (out/'predictions.jsonl').open('w') as f, (out/'errors.log').open('w') as log:
        for index,r in enumerate(rows):
            record={'id':r['id'],'status':'ok'}
            try:
                with Image.open(r['image']) as im: result=methods[r['task']](im.convert('RGB'),r['query'])
                if not result.get('full_response') and not result.get('answer'): raise RuntimeError('Empty upstream response; inspect model output')
                if 'masks' in result:
                    filename=f'{index:06d}.png'; Image.fromarray(np.asarray(result.pop('masks'),dtype=np.uint8)).save(out/filename); record['mask']=filename
                record.update(result)
            except Exception as exc:
                record.update(status='error',error=str(exc)); log.write(f"{r['id']}\n{traceback.format_exc()}\n"); log.flush()
            f.write(json.dumps(record,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n'); f.flush()
if __name__=='__main__': main()
