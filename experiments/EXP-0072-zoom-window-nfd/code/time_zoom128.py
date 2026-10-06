"""Adds the 128x128 zoom-window model (same 64 mm window at 0.5 mm/px) to results/timing.json: same method as time_zoom.py,
visual input = HxH whole-tray raster (default 300), window extraction + plates + UNet(128x128) [+ paste to the 64x64 world grid]."""
import sys, json, glob, torch
sys.path.insert(0, ".")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--H", type=int, default=300)
ap.add_argument("--json", default="experiments/EXP-0072-zoom-window-nfd/results/timing.json"); a = ap.parse_args()
import Baselines.common.benchmark_time as bt
from model.zoom_nfd.window import WindowSpec
from model.zoom_nfd.window_gpu import raster_world, extract_windows, plates_gpu, paste_gpu
from simple_mpc.adapters import occ_from_particles
import importlib.util
spec_ = importlib.util.spec_from_file_location("tz", "experiments/EXP-0072-zoom-window-nfd/code/time_zoom.py")
cont, det = bt.check_gpu_contention(); print("contended:", cont)
assert not cont, "GPU busy"
dev = "cuda"; SIZES = [[0.005] * 3] * 20; spec = WindowSpec(res=128)
from model.UNetModels_modular import UNet
import time
def unet(path):
    n = UNet({"features": [4, 8, 16], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu",
              "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}}); n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval()
def timeit(fn, k, warmup=5, repeats=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    m = bt._median_iqr(ts); return {"total_ms": m, "per_candidate_us": {kk: v * 1e3 / k if kk != "n" else v for kk, v in m.items()}}
net = unet(a.ckpt)
pl = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_pools_v2/pools_*.pt"))[0], map_location="cpu", weights_only=False)
S0 = pl["states"][0:1].float().cpu(); acts = torch.cat([pl["p_starts"][:, :2], pl["p_stops"][:, :2]], 1).float()
o0w = occ_from_particles(S0)[0].to(dev); hr = raster_world(S0, SIZES, a.H)[0].to(dev)
D = json.load(open(a.json)); name = f"zoom128_hr{a.H}"
for k in (1, 32, 128, 1024):
    A = acts[[j % len(acts) for j in range(k)]].to(dev); P0, P1 = A[:, :2], A[:, 2:]
    front = lambda: ((extract_windows(hr, P0, P1, spec) > 0.4).float(), plates_gpu(P0, P1, spec))
    fwd = lambda w, p: torch.sigmoid(net(torch.cat([w[:, None], p], 1))).squeeze(1)
    with torch.no_grad():
        w_, p_ = front(); d_ = fwd(w_, p_) - w_
        r = {"window_out": timeit(lambda: fwd(*front()), k),
             "world_out": timeit(lambda: (lambda w, p: paste_gpu(fwd(w, p) - w, o0w, P0, P1, spec))(*front()), k),
             "stages": {"extract": timeit(lambda: (extract_windows(hr, P0, P1, spec) > 0.4).float(), k), "plates": timeit(lambda: plates_gpu(P0, P1, spec), k),
                        "unet": timeit(lambda: fwd(w_, p_), k), "paste": timeit(lambda: paste_gpu(d_, o0w, P0, P1, spec), k)}}
    D["ours"][str(k)][name] = r
    print(k, round(r["window_out"]["total_ms"]["median"], 2), round(r["world_out"]["total_ms"]["median"], 2), {n: round(v["total_ms"]["median"], 2) for n, v in r["stages"].items()}, flush=True)
json.dump(D, open(a.json, "w"), indent=1)
