"""Characterise a sand dataset: is the physics sane, and is the data diverse?

Run before trusting any fit. The cube work repeatedly produced runs that
completed with plausible-looking data and wrong physics, so these are the
checks that would have caught them: mass conservation, containment, actual
grain displacement, and -- new for the varied-start dataset -- how much the
START states actually differ from each other.
"""
from __future__ import annotations
import argparse, glob as _glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transforms.sand_occupancy import sand_mass, sand_to_density, sand_to_mask

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--grid", type=int, default=64)
a = ap.parse_args()
B = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}

files = sorted(_glob.glob(a.glob, recursive=True))
S0, S1, PS, PE, EP, IDX = [], [], [], [], [], []
for i, f in enumerate(files):
    d = torch.load(f, weights_only=False)
    n = d["states"].shape[0]
    S0.append(d["states"][..., :3]); S1.append(d["states_"][..., :3])
    PS.append(d["p_starts"]); PE.append(d["p_stops"])
    EP.append(torch.full((n,), i)); IDX.append(torch.arange(n))
s0, s1 = torch.cat(S0), torch.cat(S1)
ps, pe, ep, idx = torch.cat(PS), torch.cat(PE), torch.cat(EP), torch.cat(IDX)
print(f"{len(files)} episodes, {s0.shape[0]} transitions, {s0.shape[1]} grains\n")

L = (pe[:, :2] - ps[:, :2]).norm(dim=-1) * 1000
full = L >= 19.9
print("--- actions ---")
print(f"push length: mean {float(L.mean()):.2f} sd {float(L.std()):.2f} mm; "
      f"full-length {100 * float(full.float().mean()):.1f}%")

print("\n--- physics ---")
m0, m1 = sand_mass(s0, B), sand_mass(s1, B)
print(f"mass in tray: before {float(m0.mean()):.4f} (min {float(m0.min()):.4f}) "
      f"| after {float(m1.mean()):.4f} (min {float(m1.min()):.4f})")
print(f"z range: {1000 * float(s0[..., 2].min()):.1f} .. {1000 * float(s0[..., 2].max()):.1f} mm "
      f"(tray floor 10.0)")
d = (s1 - s0)[..., :2].norm(dim=-1) * 1000
print(f"grain displacement: mean {float(d.mean()):.2f} p95 {float(d.quantile(0.95)):.2f} "
      f"max {float(d.max()):.2f} mm")

print("\n--- start-state diversity (episode starts only) ---")
starts = s0[idx < (s0.shape[0] // max(len(files), 1)) // 5 + 1] if False else None
# first transition of each episode file, i.e. every env's push 1
first = idx < (torch.bincount(ep).float().mean() / 5).item()
c = s0[first][..., :2].mean(1)
print(f"episode-start pile centroid spread: {1000 * float(c.std(0).mean()):.2f} mm")
r = (s0[first][..., :2] - c[:, None, :]).norm(dim=-1).mean(1)
print(f"episode-start pile radius: mean {1000 * float(r.mean()):.1f} sd {1000 * float(r.std()):.1f} mm")

print("\n--- observation maps ---")
for name, fn in (("density", lambda x: sand_to_density(x, B, (a.grid, a.grid), normalize=None)),
                 ("mask>=2", lambda x: sand_to_mask(x, B, (a.grid, a.grid), min_grains=2))):
    f0 = fn(s0[:512])
    print(f"{name:9s}: occupied {float((f0 > 0).float().sum(dim=(1,2)).mean()):.0f}/"
          f"{a.grid**2} cells, max {float(f0.max()):.1f}")
