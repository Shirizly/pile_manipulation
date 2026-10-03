"""EXP-0062 scratch breakdown of NFD predict_occ time (plates / forward / D2H / batch scaling) on one DS-0019 slate. Numbers quoted in runs/RUN-0001-nfd/RUN.md. Run: PYTHONPATH=. python -u experiments/EXP-0062-flex-v2-train-rerun/code/profile_nfd_breakdown.py"""
import sys,os,time,torch
sys.path.insert(0,'.'); sys.path.insert(0,'experiments/EXP-0062-flex-v2-train-rerun/code')
from Baselines.common import eval_report as er
import time_inference as ti
spec=dict(er.MODELS['nfd_randlen'],ckpt='weights/MODEL-0008-nfd-flex-mask-v2-seed0/checkpoint.pth')
cell=er._load_cell(er.CORPORA['flex_ds0019_mask'],tag='x'); p=er._load_predictor(spec)
rows=(cell.slate_idx==cell.slate_idx[0]).nonzero(as_tuple=True)[0].tolist()
b=ti.sub_batch(cell,rows,'cuda'); print('B',len(rows))
for _ in range(20): p.predict_occ(b)
from transforms.functional import draw_plate_soft
from Baselines.NFD.predictor import _plate_geometry_px
raw=b.raw; H,W=b.H,b.W
def t(f,n=50):
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): r=f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3, r
ctr=raw.ctr_in_PXL.to(torch.float32).cuda()[:2]
px,py,sg=_plate_geometry_px(raw)
def plates():
    s=b.p_start[:,:2]*raw.to_pxl+ctr; e=b.p_stop[:,:2]*raw.to_pxl+ctr
    return torch.stack([b.occ0, draw_plate_soft(s,b.angle,(H,W),px,py,1.0,sg), draw_plate_soft(e,b.angle,(H,W),px,py,1.0,sg)],1)
ms,x=t(plates); print('plates+stack ms',ms)
with torch.no_grad():
    ms,y=t(lambda: torch.sigmoid(p.model(x)).squeeze(1)); print('forward+sigmoid ms',ms)
    ms,_=t(lambda: y.cpu()); print('D2H ms',ms)
    ms,_=t(lambda: p.predict_occ(b)); print('predict_occ ms',ms)
    for bs in (16,128,512,2048):
        xx=x[:1].expand(bs,-1,-1,-1).contiguous()
        ms,_=t(lambda: p.model(xx)); print('forward only B',bs,ms)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ms,_=t(lambda: p.model(x)); print('forward bf16 autocast',ms)
print(torch.cuda.get_device_name(0))
