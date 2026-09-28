"""EXP-0046 step 1-2: goal ceiling V*(g) for 20 single-layer, non-overlapping
5 mm cubes, scored exactly as TRUE outcomes are (occ_for_scoring soft splat,
sigma 1 px) under lyapunov; plus capacity, stroke width, and whether the
lyapunov optimum also maximises mass_in_region / signed_mass.

Key fact used: occ_for_scoring is a sum of per-particle separable Gaussians,
each of the same mass, so lyapunov = mean over cubes of a per-cube cost map
c(x, y) = gu(x)^T D gv(y). V* is therefore a packing problem: choose 20
centres (0.5 mm grid, inside the tray) with pairwise Chebyshev spacing
>= 5 mm (axis-aligned squares don't overlap) minimising sum c. Solved by
greedy + 10x10 lattice offsets + coordinate-descent refinement (upper bound
on the true optimum, i.e. V* reported is achievable).

Checkpoints results/vstar.json atomically after every goal.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, minimum_filter

sys.path.insert(0, os.getcwd())
from Baselines.common.eval_report import _fixed_goal, goal_names  # noqa: E402
from Baselines.common.goals import mass_in_region, signed_mass_in_region  # noqa: E402
from simple_mpc.adapters import (OCC_BOUNDS, OCC_CUBE_SIZE, OCC_FOOTPRINT_RADIUS,  # noqa: E402
                                 OCC_GRID, occ_for_scoring)
from simple_mpc.learned_mpc import lyap  # noqa: E402

OUT = "experiments/EXP-0046-goal-ceiling/results/vstar.json"
N = 20
STEP = 0.0005
Z0 = 0.0125                        # DS-0006 layer-0 cube-centre height
HALF = OCC_CUBE_SIZE / 2
XS = np.round(np.arange(OCC_BOUNDS["x_min"] + HALF, OCC_BOUNDS["x_max"] - HALF + 1e-9, STEP), 6)
K = len(XS)
SEP = int(round(OCC_CUBE_SIZE / STEP))   # 10 grid steps = 5 mm
LO, HI = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]
PITCH_SCORE = (HI - LO) / (OCC_GRID - 1)  # scoring mapping: pixel-centre spacing


def gauss_rows():
    uv = (XS - LO) / (HI - LO) * (OCC_GRID - 1)
    i = np.arange(OCC_GRID)
    g = np.exp(-(i[None] - uv[:, None]) ** 2 / 2.0)
    return g / g.sum(1, keepdims=True)                  # (K, 64)


G = gauss_rows()


def per_cube_map(field):
    return G @ field @ G.T                              # (K, K): value of one cube at (x_i, y_j)


def conflict_box(occ, i, j, val=1):
    occ[max(0, i - SEP + 1):i + SEP, max(0, j - SEP + 1):j + SEP] += val


def greedy(C, allowed=None, n=N):
    occ = np.zeros((K, K), np.int32)
    pts = []
    Cm = C.copy() if allowed is None else np.where(allowed, C, np.inf)
    for _ in range(n):
        Cf = np.where(occ == 0, Cm, np.inf)
        k = int(np.argmin(Cf))
        if not np.isfinite(Cf.flat[k]):
            break
        i, j = divmod(k, K)
        pts.append((i, j)); conflict_box(occ, i, j)
    return pts


def lattice(C, allowed=None, n=N):
    best, bpts = np.inf, None
    Cm = C if allowed is None else np.where(allowed, C, np.inf)
    for oi in range(SEP):
        for oj in range(SEP):
            sub = Cm[oi::SEP, oj::SEP]
            flat = np.argsort(sub, axis=None)[:n]
            vals = sub.flat[flat]
            if allowed is not None:
                flat = flat[np.isfinite(vals)]
                vals = vals[np.isfinite(vals)]
                if len(flat) < n:
                    continue
            s = vals.sum()
            if s < best:
                ii, jj = np.unravel_index(flat, sub.shape)
                best, bpts = s, [(oi + SEP * a, oj + SEP * b) for a, b in zip(ii, jj)]
    return bpts


def refine(C, pts, passes=30):
    pts = list(pts)
    for _ in range(passes):
        moved = False
        for q in range(len(pts)):
            occ = np.zeros((K, K), np.int32)
            for r, (i, j) in enumerate(pts):
                if r != q:
                    conflict_box(occ, i, j)
            Cf = np.where(occ == 0, C, np.inf)
            k = int(np.argmin(Cf))
            i, j = divmod(k, K)
            if Cf[i, j] < C[pts[q]] - 1e-12:
                pts[q] = (i, j); moved = True
        if not moved:
            break
    return pts


def optimise(C):
    cands = [greedy(C)]
    lat = lattice(C)
    if lat is not None:
        cands.append(lat)
    rng = np.random.default_rng(0)
    scale = float(np.std(C)) + 1e-12
    for _ in range(12):                              # noisy-greedy restarts
        cands.append(greedy(C + rng.normal(0, 0.05 * scale, C.shape)))
    cands = [refine(C, p) for p in cands if len(p) == N]
    return min(cands, key=lambda p: sum(C[q] for q in p))


def fine_inside(mask):
    """(K,K) bool: cube footprint centred at (XS[i], XS[j]) lies entirely in
    mask pixels (pixel cells in the SCORING mapping, nearest pixel centre)."""
    f = np.round(np.arange(LO, HI + 1e-9, STEP), 6)
    idx = np.clip(np.round((f - LO) / PITCH_SCORE).astype(int), 0, OCC_GRID - 1)
    F = mask[np.ix_(idx, idx)]
    E = minimum_filter(F.astype(np.uint8), size=2 * int(round(HALF / STEP)) + 1, mode="constant", cval=0)
    off = int(round(HALF / STEP))
    return E[off:off + K, off:off + K].astype(bool)


def capacity(inside):
    best = 0
    for oi in range(SEP):
        for oj in range(SEP):
            best = max(best, int(inside[oi::SEP, oj::SEP].sum()))
    best = max(best, len(greedy(np.zeros((K, K)), allowed=inside, n=10 ** 6)))
    return best


def evaluate(pts, mask_t, dist_t):
    st = torch.tensor([[XS[i], XS[j], Z0] for i, j in pts], dtype=torch.float32)[None]
    occ = occ_for_scoring(st)
    tot = float(occ.sum())
    return dict(lyapunov=float(lyap(occ, dist_t)[0]),
                mass_in_region=float(mass_in_region(occ, mask_t)[0]),
                signed_mass=float(signed_mass_in_region(occ, mask_t)[0]),
                total_mass=tot)


def check_nonoverlap(pts):
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if max(abs(pts[a][0] - pts[b][0]), abs(pts[a][1] - pts[b][1])) < SEP:
                return False
    return True


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    total_mass = N * math.pi * OCC_FOOTPRINT_RADIUS ** 2
    for g in goal_names("many_plus"):
        if g in res:
            continue
        mask_t, dist_t = _fixed_goal(g, OCC_GRID, OCC_GRID)
        mask = mask_t.numpy() > 0
        D = dist_t.numpy().astype(np.float64)
        Cl = per_cube_map(D)
        Cm = -per_cube_map(mask.astype(np.float64))     # maximise mass-in-mask
        p_l = optimise(Cl)
        p_m = min([optimise(Cm), refine(Cm, p_l)], key=lambda p: sum(Cm[q] for q in p))
        inside = fine_inside(mask)
        edt = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]   # tray edge = outside
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(mask)
            med_w = float(2 * np.median(edt[sk]) * PITCH_SCORE * 1e3) if sk.any() else float("nan")
        except Exception:
            med_w = float("nan")
        r = dict(
            vstar=evaluate(p_l, mask_t, dist_t),
            best_mass_placement=evaluate(p_m, mask_t, dist_t),
            vstar_map_value=float(np.mean([Cl[q] for q in p_l])),
            nonoverlap_ok=check_nonoverlap(p_l) and check_nonoverlap(p_m),
            n_inside_vstar=int(sum(inside[q] for q in p_l)),
            capacity=capacity(inside),
            mask_px=int(mask.sum()),
            stroke_width_max_mm=float(2 * edt.max() * PITCH_SCORE * 1e3),
            stroke_width_median_skeleton_mm=med_w,
            theoretical_max_mass=total_mass,
            vstar_centres_m=[[float(XS[i]), float(XS[j])] for i, j in p_l],
        )
        r["all_fit"] = r["n_inside_vstar"] == N
        v = r["vstar"]
        r["mass_frac_at_vstar"] = v["mass_in_region"] / total_mass
        r["signed_frac_at_vstar"] = v["signed_mass"] / total_mass
        r["mass_frac_best_placement"] = r["best_mass_placement"]["mass_in_region"] / total_mass
        res[g] = r
        tmp = OUT + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, OUT)
        print(f"{g:12s} V*={v['lyapunov']:.4f} cap={r['capacity']:3d} in={r['n_inside_vstar']:2d} "
              f"w={r['stroke_width_max_mm']:.1f}mm massfrac={r['mass_frac_at_vstar']:.3f} "
              f"best_massfrac={r['mass_frac_best_placement']:.3f} signed={r['signed_frac_at_vstar']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
