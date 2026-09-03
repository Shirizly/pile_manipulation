"""Effective dimensionality of sand states, changes, and the fitted operator.

The rank results only mean something next to the dimensionality of the INPUT
states: an operator cannot need more rank than its inputs supply. On the
single-pile dataset the canonical input states spanned ~14 dims for 90% of
their variance, so a rank-4 operator was partly a statement about the data.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from sand_foresight import BOUNDS, load_sand_arrays
from fit_linear_foresight import actions_to_pixels, canonicalise, fit_operator

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=0.5)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

for view, blur in (("density", 0.0), ("mask", 1.0)):
    occ_t, occ_t1, actions, ep, s0, s1 = load_sand_arrays(
        a.glob, 64, blur, None, 19.9, dev, view=view, min_grains=2)
    s, e = actions_to_pixels(actions, (BOUNDS["x_min"], BOUNDS["y_min"]),
                             (BOUNDS["x_max"], BOUNDS["y_max"]), (64, 64))
    s, e = s.to(dev), e.to(dev)
    Y0 = canonicalise(occ_t, s, e, a.res, a.crop).reshape(len(occ_t), -1).T
    Y1 = canonicalise(occ_t1, s, e, a.res, a.crop).reshape(len(occ_t), -1).T
    print(f"\n=== view={view} (blur {blur}) D={Y0.shape[0]} M={Y0.shape[1]} ===")
    for name, Y in (("input states", Y0), ("outputs", Y1), ("changes", Y1 - Y0)):
        Yc = Y - Y.mean(1, keepdim=True)
        sv = torch.linalg.svdvals(Yc.float())
        en = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        print(f"  {name:13s}: dims for 90%/99% variance = "
              f"{int((en < 0.90).sum()) + 1:4d} / {int((en < 0.99).sum()) + 1:4d}")
    A = fit_operator(Y0, Y1, 1.0, toward_identity=True)
    P = A @ Y0
    sv = torch.linalg.svdvals((P - P.mean(1, keepdim=True)).float())
    en = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    print("  operator predictions, variance captured by top-r modes:")
    print("    " + "  ".join(f"r={r}:{100 * float(en[r - 1]):.1f}%"
                             for r in (1, 2, 4, 8, 16, 32, 64) if r <= len(en)))
