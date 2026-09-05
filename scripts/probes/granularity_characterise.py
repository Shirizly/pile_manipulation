"""Characterise the granularity series before any model is trained.

Two properties decide whether each cell can answer anything: how much material
the blade actually engages, and whether the push length really is constant.
Pile-aware placement can put the blade where less than the requested travel
fits inside the workspace, which silently turns a fixed-length dataset into a
variable-length one.
"""
import glob, torch

SETS = [("blind  n=5",  "Genesis/data/granularity/n5/**/*_data.pt"),
        ("blind  n=10", "Genesis/data/granularity/n10/**/*_data.pt"),
        ("blind  n=20", "Genesis/data/granularity/n20/**/*_data.pt"),
        ("contact n=5",  "Genesis/data/granularity/c5/**/*_data.pt"),
        ("contact n=10", "Genesis/data/granularity/c10/**/*_data.pt"),
        ("contact n=20", "Genesis/data/granularity/c20/**/*_data.pt"),
        ("L040   n=50", "Genesis/data/foresight/L040/**/*_data.pt")]

print(f"{'dataset':14s} {'n':>6s} {'len mm':>14s} {'full-len':>9s} "
      f"{'swath':>7s} {'0-contact':>10s} {'moved>2mm':>10s}")
print("-" * 78)
for name, pat in SETS:
    fs = sorted(glob.glob(pat, recursive=True))
    if not fs:
        print(f"{name:14s}  (absent)"); continue
    L, inb, mv = [], [], []
    for f in fs:
        d = torch.load(f, map_location="cpu", weights_only=False)
        s0, s1 = d["states"][..., :2], d["states_"][..., :2]
        ps, pe = d["p_starts"][:, :2], d["p_stops"][:, :2]
        seg = pe - ps; ln = seg.norm(dim=-1, keepdim=True)
        u = seg / ln.clamp_min(1e-9)
        rel = s0 - ps[:, None, :]
        t = (rel * u[:, None, :]).sum(-1)
        perp = (rel - t.unsqueeze(-1) * u[:, None, :]).norm(dim=-1)
        inb.append(((t > -0.005) & (t < ln + 0.005) & (perp < 0.022)).sum(1))
        mv.append(((s1 - s0).norm(dim=-1) > 0.002).sum(1))
        L.append(ln.squeeze(-1))
    L, inb, mv = torch.cat(L) * 1000, torch.cat(inb).float(), torch.cat(mv).float()
    full = float((L > L.max() * 0.98).float().mean())
    print(f"{name:14s} {len(L):6d} {L.mean():7.1f}+-{L.std():5.2f} {full:8.0%} "
          f"{inb.mean():7.2f} {float((inb == 0).float().mean()):9.0%} "
          f"{mv.mean():10.2f}")
