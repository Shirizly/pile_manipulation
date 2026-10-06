import sys, time, numpy as np, torch
sys.path.insert(0, ".")
from Baselines.NFD.nfd_lib import PileSweepData3Ch
from model.zoom_nfd.world128 import raster128, plates128
out = {}
for split in ("train", "val"):
    t = time.time()
    ds = PileSweepData3Ch(paths=["narrow_l20_n20/train_v2"], split=split, resolution_scale=0.5, exclude_flagged=True, min_push_length_m=1e-4)
    sizes = ds.configs[0]["data_collection"]["sampled"]["particle_sizes"]
    S, S_, P0, P1 = [], [], [], []
    for i in range(len(ds)):
        ri = ds._resolve_idx(i); r = ds._run_lookup[ri]; k = ri - ds._offsets[r]; run = ds.runs[r]
        S.append(run["states"][k]); S_.append(run["states_"][k]); P0.append(run["p_starts"][k][:2].numpy()); P1.append(run["p_stops"][k][:2].numpy())
    S, S_ = torch.stack(S), torch.stack(S_); P0, P1 = np.stack(P0), np.stack(P1)
    x = torch.cat([raster128(S, sizes)[:, None], plates128(P0, P1)], 1).half(); y = raster128(S_, sizes).half()
    out[split] = dict(x=x, y=y); print(split, x.shape, f"{time.time()-t:.0f}s", flush=True)
torch.save(out, "experiments/EXP-0072-zoom-window-nfd/runs/world128_cache.pt")
