"""Accuracy CAP of the pasted 64x64 frame on the wide test rows: paste the TRUE next-state window (pose-rendered, 128 px, same box rasteriser as training) instead of a prediction, i.e. a perfect window prediction.
Also the same thing in a finer 128x128 WORLD frame truth (box footprints at 1 mm/px) to show how the cap depends on the evaluation raster."""
import sys, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from eval_wide import *
from model.zoom_nfd.world_res import raster_res
torch.set_num_threads(4)
sets = test_sets(); out = {}
for sh in SHARDS:
    s = sets[sh]; t = s["t"]; rows = torch.from_numpy(s["rows"])[:500]; A64, A64b, A128 = [], [], []
    for i in range(0, len(rows), 100):
        ix = rows[i:i + 100]; S0, S1, P0, P1 = t["S"][ix], t["S_"][ix], t["P0"][ix], t["P1"][ix]; n = S0.shape[1]; sz = [[0.005] * 3] * n
        X = window_batch_var(S0, sz, P0.numpy(), P1.numpy(), 128).float(); Y = window_batch_var(S1, sz, P0.numpy(), P1.numpy(), 128).float()
        side = side_for((P1 - P0).norm(dim=-1)); d = delta_to_world64_v((Y - X).cuda(), P0.cuda(), P1.cuda(), side.cuda()).cpu()
        O0, T = occ_from_particles(S0), occ_from_particles(S1); Rg = swept_region(torch.cat([P0, P1], 1), "cpu").cpu().float()
        pasted = (O0 + d).clamp(0, 1); A64.append((((pasted - T) ** 2 * Rg).sum(), ((O0 - T) ** 2 * Rg).sum()))
        # same cap if the world truth is the box raster at 64 px (same drawing style as the predictions) -> style mismatch removed, only resampling left
        B0, B1 = raster_res(S0, sz, 64), raster_res(S1, sz, 64); pb = (B0 + d).clamp(0, 1); A64b.append((((pb - B1) ** 2 * Rg).sum(), ((B0 - B1) ** 2 * Rg).sum()))
    f = lambda L: 1 - float(np.sqrt(sum(float(a) for a, _ in L) / sum(float(b) for _, b in L)))
    out[sh] = (f(A64), f(A64b)); print(sh, "cap vs occ_from_particles (disc) truth %.3f | vs box-raster truth at the same 64 px %.3f" % out[sh], flush=True)
print("MEAN", np.mean([v[0] for v in out.values()]), np.mean([v[1] for v in out.values()]))
