"""Candidate-pool timing (benchmark_time.py method: ONE state, K candidate pushes, K in {1,32,128,1024}, 5 warm-up, 15 repeats, cuda.synchronize, median [IQR]) of
the wide-domain models from a VISUAL input on the GPU (a 300x300 / world raster): zoom (variable-side window extraction + plates + UNet [+ paste to the world 64 grid])
and vanilla world NFDs (plates + UNet [+ down to 64]). usage: time_wide.py --zoom64 CKPT FEATS --zoom128 CKPT FEATS --world64 .. --world128 .. [--force]"""
import sys, json, time, glob, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import Baselines.common.benchmark_time as bt
from model.UNetModels_modular import UNet
from model.zoom_nfd.window_var import *
from model.zoom_nfd.window_var import _coords
from model.zoom_nfd.world_res import plates_res
from model.zoom_nfd.rollout import canvas_from_particles
from eval_wide import down_world
dev = "cuda"; KS = [1, 32, 128, 1024]
ap = argparse.ArgumentParser()
for n in ("zoom64", "zoom128", "world64", "world128"): ap.add_argument("--" + n, nargs=2, metavar=("CKPT", "FEATS"))
ap.add_argument("--out", default="experiments/EXP-0074-wide-domain-zoom-nfd/results/timing.json"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
cont, _ = bt.check_gpu_contention()
if cont and not a.force: sys.exit("GPU contended")
def net(path, feats):
    n = UNet({"features": [int(x) for x in feats.split(",")], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu", "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
    n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval(), sum(p.numel() for p in n.parameters())
def timeit(fn, k, warmup=5, repeats=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    m = bt._median_iqr(ts); return {"total_ms": m, "per_candidate_us": {kk: v * 1e3 / k if kk != "n" else v for kk, v in m.items()}}
def extract_same(canvas, P0, P1, side, res, ss=3):   # one state, K pushes: one grid_sample with the K windows stacked along the grid rows
    K = len(P0); C = canvas.shape[-1]; pxc = SIDE / C
    idx = (_coords(P0, P1, side, res, ss, canvas.device) - LO) / pxc - 0.5
    g = (torch.stack([idx[..., 1], idx[..., 0]], -1) / (C - 1) * 2 - 1).reshape(1, K * res * ss, res * ss, 2)
    return F.avg_pool2d(F.grid_sample(canvas[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(K, 1, res * ss, res * ss), ss).squeeze(1)
pl = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_pools_v2/pools_*.pt"))[0], map_location="cpu", weights_only=False)
S0 = pl["states"][0:1].float().cpu(); d0 = pl["p_stops"][:, :2] - pl["p_starts"][:, :2]
rng = np.random.default_rng(0)
res = {}; canvas = canvas_from_particles(S0, [[0.005] * 3] * 20, 300, 0.2)[0].float().to(dev)
from model.zoom_nfd.world_res import raster_res
for name in ("zoom64", "zoom128", "world64", "world128"):
    spec = getattr(a, name)
    if not spec: continue
    r = int(name[-2:]) if name.endswith("64") else 128; n, npar = net(*spec); res[name] = dict(params=npar, by_k={})
    st = raster_res(S0, [[0.005] * 3] * 20, r)[0].to(dev) if name.startswith("world") else None
    for k in KS:
        L = torch.from_numpy(rng.uniform(0.02, 0.07, k)).float(); ang = torch.from_numpy(rng.uniform(0, 6.28, k)).float()
        P0 = torch.from_numpy(rng.uniform(-0.04, 0.04, (k, 2))).float().to(dev); P1 = P0 + torch.stack([torch.cos(ang), torch.sin(ang)], 1).to(dev) * L[:, None].to(dev); side = side_for((P1 - P0).norm(dim=-1))
        o0 = torch.zeros(64, 64, device=dev)
        with torch.no_grad():
            if name.startswith("zoom"):
                front = lambda: (extract_same(canvas, P0, P1, side, r), plates_v(P0, P1, side, r)); fwd = lambda w, p: torch.sigmoid(n(torch.cat([w[:, None], p], 1))).squeeze(1)
                w_, p_ = front(); d_ = fwd(w_, p_) - w_
                e = {"window_out": timeit(lambda: fwd(*front()), k), "world_out": timeit(lambda: (lambda w, p: (o0[None] + delta_to_world64_v(fwd(w, p) - w, P0, P1, side)).clamp(0, 1))(*front()), k),
                     "stages": {"extract": timeit(lambda: extract_same(canvas, P0, P1, side, r), k), "plates": timeit(lambda: plates_v(P0, P1, side, r), k), "unet": timeit(lambda: fwd(w_, p_), k),
                                "paste": timeit(lambda: delta_to_world64_v(d_, P0, P1, side), k)}}
            else:
                fwd = lambda: torch.sigmoid(n(torch.cat([st[None].expand(k, -1, -1)[:, None], plates_res(P0, P1, r)], 1))).squeeze(1)
                e = {"native_out": timeit(fwd, k), "world_out": timeit(lambda: (o0[None] + down_world(fwd() - st[None], r)).clamp(0, 1), k)}
        res[name]["by_k"][str(k)] = e; print(name, k, {m: round(v["total_ms"]["median"], 2) for m, v in e.items() if m != "stages"}, flush=True)
json.dump(dict(res=res, gpu=torch.cuda.get_device_name(0), contended=cont), open(a.out, "w"), indent=1)
