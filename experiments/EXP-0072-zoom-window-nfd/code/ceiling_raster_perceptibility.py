"""Ceiling step 1a: how large a random cube-pose perturbation changes each model's INPUT raster barely/not at all.
Rasters: world64 hard (simple_mpc.adapters.occ_from_particles, 2 mm/px), world128 (1 mm/px), zoom64 window (1 mm/px), zoom128 window (0.5 mm/px).
Perturbation = iid gaussian xy (std sigma mm) + yaw jitter 2 deg/mm. 200 DS-0016 rows (every 4th), 2 independent draws each."""
import json, sys, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_common import *
from simple_mpc.adapters import occ_from_particles
from model.zoom_nfd.window import WindowSpec, window_batch
from model.zoom_nfd.world128 import raster128

c = load_chains(); ix = torch.arange(0, len(c["states"]), 4)[:200]
S = c["states"][ix]; P0 = c["p_starts"][ix, :2].numpy(); P1 = c["p_stops"][ix, :2].numpy()
R = {"world64(2mm/px)": lambda s: occ_from_particles(s).cpu(), "world128(1mm/px)": lambda s: raster128(s, SIZES),
     "zoom64(1mm/px)": lambda s: window_batch(s, SIZES, P0, P1, WindowSpec(res=64)),
     "zoom128(0.5mm/px)": lambda s: window_batch(s, SIZES, P0, P1, WindowSpec(res=128))}
sig = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]; rng = np.random.default_rng(0); res = {}
base = {k: f(S) for k, f in R.items()}
for k, f in R.items():
    res[k] = {}
    for s in sig:
        ident, chg, frac, soft = [], [], [], []
        for rep in range(2):
            Sp = perturb(S, s, rng); X = f(Sp); B = base[k]
            d = ((X > 0.5) != (B > 0.5)).flatten(1).sum(1).float(); occ = (B > 0.5).flatten(1).sum(1).float()
            ident += (d == 0).float().tolist(); chg += d.tolist(); frac += (d / occ).tolist()
            soft += ((X - B).abs().flatten(1).sum(1) / occ).tolist()      # sum|dX| / occupied px (raw hard values; = chg for hard rasters)
        res[k][str(s)] = {"frac_identical": float(np.mean(ident)), "mean_changed_px": float(np.mean(chg)),
                          "median_changed_px": float(np.median(chg)), "changed_frac_of_occupied": float(np.mean(frac))}
    print(k, flush=True)
    for s in sig: print(f"  {s:5.2f} mm  identical {res[k][str(s)]['frac_identical']:.2f}  changed px {res[k][str(s)]['mean_changed_px']:6.1f}  frac of occupied {res[k][str(s)]['changed_frac_of_occupied']:.4f}", flush=True)
json.dump(res, open(OUT + "raster_perceptibility.json", "w"), indent=1)
