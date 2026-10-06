"""Candidate-pool timing (Baselines/common/benchmark_time.py method: ONE state, K candidate pushes, K in {1,32,128,1024},
5 warm-up + 15 repeats, cuda.synchronize either side, median [IQR]) of the zoom NFD and the 128x128 NFD, next to the
reference predictors re-timed on this machine/session (original NFD, GNN, linear) with the harness' own batch & predictors.

Zoom / 128 models start from a VISUAL input on the GPU (a binary top-down raster of the tray, as a camera would give),
never from particle poses:
  zoom_hrH : HxH raster -> extract_windows (rotation+zoom, 3x3 antialiased) [+ binarise] -> plates -> UNet -> window pred
             (+ paste_gpu = change sampled back onto the 64x64 world grid, `world_out`)
  world128 : 128x128 raster -> plates -> UNet -> 128x128 pred (+ resample the change to the 64x64 world grid)
  nfd64    : this repo's 64x64 narrow NFD via its OCC adapter (raster in, raster out)
Reported: total ms per call and per candidate, with and without the paste-back, plus the zoom stage split.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
os.environ.setdefault("NFD_CKPT", "Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth")
import Baselines.common.benchmark_time as bt
from model.zoom_nfd.window import WindowSpec
from model.zoom_nfd.window_gpu import raster_world, extract_windows, plates_gpu, paste_gpu, LO, SIDE
from model.UNetModels_modular import UNet
from simple_mpc.adapters import make_occ_adapter, occ_from_particles
import torch.nn.functional as F
from transforms.functional import draw_plate_soft

dev = "cuda"; spec = WindowSpec(); SIZES = [[0.005] * 3] * 20
KS = [1, 32, 128, 1024]


def unet(path):
    n = UNet({"features": [4, 8, 16], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1,
              "activation": "relu", "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
    n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval()


def plates128_gpu(P0, P1, res=128, plate_len=0.04, thick=0.002, sigma=0.0015):
    px = SIDE / res; K = len(P0)
    ang = torch.atan2(P1[:, 1] - P0[:, 1], P1[:, 0] - P0[:, 0]) - math.pi / 2
    ch = [draw_plate_soft((P - LO) / px - 0.5, ang, (res, res), plate_len / px, thick / px, 1.0, sigma / px) for P in (P0, P1)]
    return torch.stack(ch, 1)


def downsample_delta(delta, o0, grid=64, sub=3):
    """(K,res,res) change on a res-grid over the tray -> world 64 grid (same sampling as zoom paste)."""
    res = delta.shape[-1]; pitch = SIDE / (grid - 1)
    ax = torch.arange(grid, device=dev) * pitch + LO
    off = (torch.arange(sub, device=dev) - (sub - 1) / 2) * pitch / sub
    X = (ax[:, None] + off[None, :]).reshape(-1)
    pts = torch.stack(torch.meshgrid(X, X, indexing="ij"), -1)
    ij = (pts - LO) / (SIDE / res) - 0.5
    g = (torch.stack([ij[..., 1], ij[..., 0]], -1) / (res - 1) * 2 - 1)[None].expand(len(delta), -1, -1, -1)
    s = F.avg_pool2d(F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True), sub).squeeze(1)
    return (o0[None] + s).clamp(0, 1)


def timeit(fn, k, warmup=5, repeats=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    m = bt._median_iqr(ts); return {"total_ms": m, "per_candidate_us": {kk: v * 1e3 / k if kk != "n" else v for kk, v in m.items()}}


ap = argparse.ArgumentParser()
ap.add_argument("--zoom", default="experiments/EXP-0072-zoom-window-nfd/runs/ft100/unet_best.pth")
ap.add_argument("--w128", default="experiments/EXP-0072-zoom-window-nfd/runs/world128/unet_best.pth")
ap.add_argument("--out", default="experiments/EXP-0072-zoom-window-nfd/results/timing.json")
ap.add_argument("--skip-baselines", action="store_true"); ap.add_argument("--baselines-only", action="store_true"); ap.add_argument("--force", action="store_true")
a = ap.parse_args()
cont, det = bt.check_gpu_contention()
print("contended:", cont, det.get("other_training_processes"))
if cont and not a.force: sys.exit("GPU contended; rerun idle (or --force for provisional numbers)")

# one state, K candidate pushes from DS-0016 pool 0
pl = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_pools_v2/pools_*.pt"))[0], map_location="cpu", weights_only=False)
S0 = pl["states"][0:1].float().cpu()
acts = torch.cat([pl["p_starts"][:, :2], pl["p_stops"][:, :2]], 1).float().cpu()
res = {}
if a.baselines_only: res = json.load(open(a.out))['ours']
znet, wnet = unet(a.zoom), unet(a.w128)
ad64 = make_occ_adapter("nfd_3ch_narrow_l20_v2", dev, "corner")
o0w = occ_from_particles(S0)[0].to(dev)
HR = {H: raster_world(S0, SIZES, H)[0].to(dev) for H in (128, 256, 400)}
for k in ([] if a.baselines_only else KS):
    A = acts[[j % len(acts) for j in range(k)]].to(dev); P0, P1 = A[:, :2], A[:, 2:]
    r = {}
    for H in (256, 400):
        def front():
            w = (extract_windows(HR[H], P0, P1, spec) > 0.2).float()
            return w, plates_gpu(P0, P1, spec)
        def fwd(w, p): return torch.sigmoid(znet(torch.cat([w[:, None], p], 1))).squeeze(1)
        @torch.no_grad()
        def win_out(): w, p = front(); return fwd(w, p)
        @torch.no_grad()
        def world_out(): w, p = front(); return paste_gpu(fwd(w, p) - w, o0w, P0, P1, spec)
        @torch.no_grad()
        def s_ext(): return (extract_windows(HR[H], P0, P1, spec) > 0.2).float()
        w_, p_ = front()
        @torch.no_grad()
        def s_plate(): return plates_gpu(P0, P1, spec)
        @torch.no_grad()
        def s_fwd(): return fwd(w_, p_)
        d_ = fwd(w_, p_) - w_
        @torch.no_grad()
        def s_paste(): return paste_gpu(d_, o0w, P0, P1, spec)
        r[f"zoom_hr{H}"] = {"window_out": timeit(win_out, k), "world_out": timeit(world_out, k),
                            "stages": {n: timeit(f, k) for n, f in (("extract", s_ext), ("plates", s_plate), ("unet", s_fwd), ("paste", s_paste))}}
    hr = HR[128]
    @torch.no_grad()
    def w128_native():
        x = torch.cat([hr[None].expand(k, -1, -1)[:, None], plates128_gpu(P0, P1)], 1)
        return torch.sigmoid(wnet(x)).squeeze(1)
    @torch.no_grad()
    def w128_world():
        x = torch.cat([hr[None].expand(k, -1, -1)[:, None], plates128_gpu(P0, P1)], 1)
        return downsample_delta(torch.sigmoid(wnet(x)).squeeze(1) - hr[None], o0w)
    r["world128"] = {"native_out": timeit(w128_native, k), "world_out": timeit(w128_world, k)}
    occ = o0w[None].expand(k, -1, -1).contiguous()
    @torch.no_grad()
    def nfd64(): return ad64.predict_step(occ, A)
    r["nfd64_narrow(adapter)"] = {"world_out": timeit(nfd64, k)}
    res[str(k)] = r
    print("K", k, {m: {kk: round(vv["total_ms"]["median"], 2) for kk, vv in v.items() if kk != "stages"} for m, v in r.items()}, flush=True)

json.dump({"ours": res, "baselines": {}, "contended": cont}, open(a.out, "w"), indent=1) if not a.baselines_only else None
base = {}
if not a.skip_baselines:
    cell = bt.load_cell(bt.DEFAULT_EVAL_CFG, "train")
    preds = [p for p, _, e in bt.load_extra_predictors() if p is not None and p.name in ("gnn", "nfd_unet3ch")]
    preds += bt.build_reference_predictors(bt.DEFAULT_TRAIN_CFG)
    for p in preds:
        base[p.name] = {}
        for k in KS:
            b = bt.build_candidate_batch(cell, k, dev)
            try: bt.time_predictor_at_k(p, b, k, 1, 1, True)
            except TypeError: b = bt.build_candidate_batch(cell, k, 'cpu')   # GNN/linear predictors take host tensors
            base[p.name][str(k)] = bt.time_predictor_at_k(p, b, k, 5, 15, True); print("base", p.name, k, round(base[p.name][str(k)]["total_ms"]["median"], 2), flush=True)
        base[p.name]["device"] = bt.actual_device_of(p); base[p.name]["params"] = bt.param_count_of(p)
json.dump({"ours": res, "baselines": base, "params": {"zoom/64-unet": sum(p.numel() for p in znet.parameters())},
           "contended": cont, "gpu": torch.cuda.get_device_name(0)}, open(a.out, "w"), indent=1)
print("wrote", a.out)
