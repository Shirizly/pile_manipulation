"""ISS-010 legality audit extended to DS-0001 / DS-0006 (pre-fix pile-aware sampler). Reuses
EXP-0059 audit_tool_placement._row_illegal (exact SAT, blade 40x2 mm vs 5 mm cubes). Writes JSON only."""
import json, sys
import numpy as np, torch
sys.path.insert(0, 'experiments/EXP-0059-retrieval-transition-model/code')
from audit_tool_placement import _row_illegal, MARGIN
out = {}
for name, p in [('DS-0001', 'Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm/step0.pt'),
                ('DS-0006', 'Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/step0.pt')]:
    d = torch.load(p, map_location='cpu', weights_only=False)
    s0 = d['states'].float(); ps = d['p_starts'].float()[:, :2]; an = d['angles'].float()
    i0, _ = _row_illegal(s0, ps, an, tol=0.0); i1, _ = _row_illegal(s0, ps, an, tol=-MARGIN)
    sl = d['slate_idx'].numpy()
    per = np.array([i0[sl == k].mean() for k in np.unique(sl)])
    out[name] = dict(n=int(len(i0)), frac_illegal_0mm=float(i0.mean()), frac_illegal_1mm=float(i1.mean()),
                     per_slate_frac_min=float(per.min()), per_slate_frac_median=float(np.median(per)), per_slate_frac_max=float(per.max()))
    print(name, out[name], flush=True)
json.dump(out, open('experiments/EXP-0065-legality-audit-ds0001-ds0006/results/audit_ds0001_ds0006.json', 'w'), indent=1)
