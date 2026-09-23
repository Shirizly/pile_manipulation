"""RUN-0001 part C -- is EXP-0015's Fourier-order reversal caused by a leak in
the inner ridge-lambda CV?

MECHANISM AS READ FROM THE CODE (verified, not assumed):
  experiments/EXP-0015-.../code/stage0_upper_bound.py::fit_eval_cv does
      folds = np.array_split(rng_cv.permutation(n), 3)
  i.e. the inner folds are ROW-RANDOM, while the OUTER split (split_state) is
  slate-aware. Post-push states inside one slate share a pile, so a row-random
  inner fold puts near-duplicates on both sides and should select too small a
  lambda; the penalty grows with descriptor dimension, which is what a
  monotone-worse Fourier sweep looks like.

TEST: same sweep, inner folds grouped by SLATE vs row-random. If the reversal
disappears under slate-aware folds, the mechanism is CONFIRMED.
Reduced grid vs stage0 (2 goals x 2 value fns) for budget.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
from control_utility_test import lyapunov                        # noqa: E402
from dmdc_baseline import occupancy_descriptors                   # noqa: E402
from transforms.functional import particles_to_occupancy          # noqa: E402
from Baselines.common.goals import (quadrant_mask, letter_mask,   # noqa: E402
                                     dist_field_from_mask, mass_in_region)
from Genesis.binned_slate_dataset import BinnedSlateCorpus        # noqa: E402
from utils import git_provenance                                  # noqa: E402

ART = EXP / "artifacts/RUN-0001"; ART.mkdir(parents=True, exist_ok=True)
CORPUS = REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID, CUBE = 64, 0.005
RADIUS = 0.5 * CUBE / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID)
PER_SLATE = 200
NF_GRID = [8, 16, 24, 32]
LAM_GRID = [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]
GOALS = ["quadrant0", "letter_T"]
VFNS = ["lyapunov", "mass_in_region"]
N_REPS = 3

def ridge_fit(X, y, lam):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return {"mu": mu, "sd": sd, "w": w, "b0": y.mean()}

def ridge_predict(m, X):
    return m["b0"] + ((X - m["mu"]) / m["sd"]) @ m["w"]

def r2(yt, yp, ref=None):
    ref = yt.mean() if ref is None else ref
    ss_tot = np.sum((yt - ref) ** 2)
    return float("nan") if ss_tot < 1e-12 else 1 - np.sum((yt - yp) ** 2) / ss_tot

def folds_row_random(n, groups, rng):
    return np.array_split(rng.permutation(n), 3)

def folds_slate_aware(n, groups, rng):
    g = np.unique(groups); rng.shuffle(g)
    parts = np.array_split(g, 3)
    return [np.nonzero(np.isin(groups, p))[0] for p in parts]

def fit_eval(Xtr, ytr, Xte, yte, groups_tr, fold_fn, seed):
    rng = np.random.default_rng(seed)
    folds = fold_fn(len(ytr), groups_tr, rng)
    best_lam, best = LAM_GRID[0], -np.inf
    for lam in LAM_GRID:
        sc = []
        for k in range(3):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(3) if j != k])
            m = ridge_fit(Xtr[tr], ytr[tr], lam)
            sc.append(r2(ytr[te], ridge_predict(m, Xtr[te])))
        s = float(np.mean(sc))
        if s > best: best, best_lam = s, lam
    m = ridge_fit(Xtr, ytr, best_lam)
    return r2(yte, ridge_predict(m, Xte), ref=ytr.mean()), best_lam

print("loading corpus...", flush=True)
corpus = BinnedSlateCorpus.load(str(CORPUS))
rows = corpus.step(0)
sidx = rows.slate_idx.numpy()
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.nonzero(sidx == s)[0],
                                 size=min(PER_SLATE, int((sidx == s).sum())), replace=False)
                      for s in range(corpus.n_slates)]); idx.sort()
states_ = rows.states_[idx].float()
slate_of_state = sidx[idx]
print(f"S={len(idx)}", flush=True)
occ = torch.cat([particles_to_occupancy(states_[a:a+1000, :, :3], BOUNDS, (GRID, GRID),
                                        footprint_radius=RADIUS)
                 for a in range(0, len(idx), 1000)])
masks = {"quadrant0": quadrant_mask(GRID, GRID, 0), "letter_T": letter_mask("T", GRID, GRID)}
vals = {}
for g in GOALS:
    dfield = torch.from_numpy(dist_field_from_mask(masks[g]))
    mt = torch.from_numpy(masks[g].astype(np.float32))
    vals[g] = {"lyapunov": lyapunov(occ, dfield).numpy().astype(np.float64),
               "mass_in_region": mass_in_region(occ, mt).numpy().astype(np.float64)}

def split_state(seed):
    r = np.random.default_rng(seed)
    sl = np.arange(int(slate_of_state.max()) + 1); r.shuffle(sl)
    te = set(sl[:4].tolist())
    return np.array([s not in te for s in slate_of_state])

out = []
for nf in NF_GRID:
    phi = occupancy_descriptors(occ, n_fourier=nf).numpy().astype(np.float64)
    for g in GOALS:
        for v in VFNS:
            y = vals[g][v]
            for name, fn in (("row_random", folds_row_random), ("slate_aware", folds_slate_aware)):
                rs, lams = [], []
                for rep in range(N_REPS):
                    trm = split_state(rep)
                    a, lam = fit_eval(phi[trm], y[trm], phi[~trm], y[~trm],
                                      slate_of_state[trm], fn, seed=1000 + rep)
                    rs.append(a); lams.append(lam)
                out.append({"n_fourier": nf, "dim": phi.shape[1], "goal": g, "value_fn": v,
                            "inner_cv": name, "r2_mean": float(np.mean(rs)),
                            "r2_std": float(np.std(rs)), "lams": lams})
                print(f"  nf={nf:2d} D={phi.shape[1]:3d} {g:10s} {v:16s} {name:11s} "
                      f"R2={np.mean(rs):+.3f}+-{np.std(rs):.3f} lam={lams}", flush=True)
                json.dump(out, open(ART / "cv_leak.json", "w"), indent=1)
json.dump({"rows": out, "provenance": git_provenance()}, open(ART / "cv_leak.json", "w"), indent=1)
print("done", flush=True)
