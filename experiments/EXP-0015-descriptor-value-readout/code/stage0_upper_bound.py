"""Stage 0 — upper-bound probe: how well is value(state, goal) recoverable
from the TRUE descriptor of the TRUE state, with the goal held FIXED?

No goal-configuration generator, no model predictions — just
occupancy_descriptors(true state occ) -> ridge -> value, for a handful of
fixed goal shapes x 3 value functions x a Fourier-order sweep (8,16,24,32).

Reuses the exact corpus / subsample / split protocol from
experiments/temp/desc-value-regression/{run,fit}.py so state-split R^2 at
n_fourier=8 should reproduce that run's per-goal numbers as a sanity check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from control_utility_test import lyapunov                       # noqa: E402
from dmdc_baseline import occupancy_descriptors                  # noqa: E402
from transforms.functional import particles_to_occupancy         # noqa: E402
from Baselines.common.goals import (                             # noqa: E402
    quadrant_mask, letter_mask, dist_field_from_mask,
    mass_in_region, signed_mass_in_region,
)
from Genesis.binned_slate_dataset import BinnedSlateCorpus       # noqa: E402
from utils import git_provenance                                 # noqa: E402

OUT = Path(__file__).resolve().parents[1]  # experiments/temp/desc-value-readout
CORPUS = REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
DEVICE = "cpu"
PER_SLATE_SAMPLES = 200
SEED = 0
N_FOURIER_GRID = [8, 16, 24, 32]
GOAL_NAMES = ["quadrant0", "quadrant1", "letter_T", "letter_O"]
VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
LAM_GRID = [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]

rng_cv = np.random.default_rng(0)


def build_masks():
    return {
        "quadrant0": quadrant_mask(GRID, GRID, 0),
        "quadrant1": quadrant_mask(GRID, GRID, 1),
        "letter_T": letter_mask("T", GRID, GRID),
        "letter_O": letter_mask("O", GRID, GRID),
    }


def occ_of(states):
    return particles_to_occupancy(states[..., :3].to(DEVICE), BOUNDS, (GRID, GRID),
                                   footprint_radius=RADIUS)


def chunked(fn, n, size=1000):
    return torch.cat([fn(lo, min(lo + size, n)) for lo in range(0, n, size)])


def ridge_fit(Xtr, ytr, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
    Xs = (Xtr - mu) / sd
    f = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(f)
    b = Xs.T @ ytr
    w = np.linalg.solve(A, b)
    return {"mu": mu, "sd": sd, "w": w, "b0": ytr.mean()}


def ridge_predict(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    return model["b0"] + Xs @ model["w"]


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if ss_tot < 1e-12 else 1 - ss_res / ss_tot


def fit_eval_cv(Xtr, ytr, Xte, yte, lam_grid):
    n = Xtr.shape[0]
    folds = np.array_split(rng_cv.permutation(n), 3)
    best_lam, best_score = lam_grid[0], -np.inf
    for lam in lam_grid:
        scores = []
        for k in range(3):
            te_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(3) if j != k])
            m = ridge_fit(Xtr[tr_idx], ytr[tr_idx], lam)
            scores.append(r2(ytr[te_idx], ridge_predict(m, Xtr[te_idx])))
        s = np.mean(scores)
        if s > best_score:
            best_score, best_lam = s, lam
    model = ridge_fit(Xtr, ytr, best_lam)
    pred = ridge_predict(model, Xte)
    return {"lam": best_lam, "r2": r2(yte, pred), "rmse": float(np.sqrt(np.mean((yte - pred) ** 2)))}


def split_state(slate_of_state, n_test_slates=4, seed=0):
    r = np.random.default_rng(seed)
    slates = np.arange(int(slate_of_state.max()) + 1)
    r.shuffle(slates)
    test_slates = set(slates[:n_test_slates].tolist())
    is_test = np.array([s in test_slates for s in slate_of_state])
    return ~is_test, is_test


def main():
    print("loading corpus...", flush=True)
    corpus = BinnedSlateCorpus.load(str(CORPUS))
    rows = corpus.step(0)
    n_slates = corpus.n_slates
    slate_idx_np = rows.slate_idx.numpy()

    rng = np.random.default_rng(SEED)
    idx_list = []
    for s in range(n_slates):
        rows_s = np.nonzero(slate_idx_np == s)[0]
        take = min(PER_SLATE_SAMPLES, len(rows_s))
        idx_list.append(rng.choice(rows_s, size=take, replace=False))
    idx = np.concatenate(idx_list); idx.sort()
    states_ = rows.states_[idx].float()
    slate_of_state = rows.slate_idx[idx].numpy()
    print(f"S={len(idx)} states", flush=True)

    print("rasterising state occupancies...", flush=True)
    occ_state = chunked(lambda a, b: occ_of(states_[a:b]), len(idx))

    masks = build_masks()
    dist_fields = {g: torch.from_numpy(dist_field_from_mask(masks[g])) for g in GOAL_NAMES}
    mask_t = {g: torch.from_numpy(masks[g].astype(np.float32)) for g in GOAL_NAMES}

    # value targets per (goal, value_fn), independent of n_fourier
    values = {}
    for g in GOAL_NAMES:
        values[g] = {
            "lyapunov": lyapunov(occ_state, dist_fields[g]).numpy().astype(np.float64),
            "mass_in_region": mass_in_region(occ_state, mask_t[g]).numpy().astype(np.float64),
            "signed_mass_in_region": signed_mass_in_region(occ_state, mask_t[g]).numpy().astype(np.float64),
        }

    results = {}  # results[nf][goal][value_fn] = {r2_mean, r2_std, lam}
    for nf in N_FOURIER_GRID:
        print(f"\n=== n_fourier={nf} ===", flush=True)
        phi = occupancy_descriptors(occ_state, n_fourier=nf).numpy().astype(np.float64)
        D = phi.shape[1]
        results[nf] = {"dim": D, "goals": {}}
        for g in GOAL_NAMES:
            results[nf]["goals"][g] = {}
            for vname in VALUE_NAMES:
                y = values[g][vname]
                r2s = []
                for rep in range(3):
                    tr_mask, te_mask = split_state(slate_of_state, seed=rep)
                    res = fit_eval_cv(phi[tr_mask], y[tr_mask], phi[te_mask], y[te_mask], LAM_GRID)
                    r2s.append(res["r2"])
                r2_mean, r2_std = float(np.mean(r2s)), float(np.std(r2s))
                results[nf]["goals"][g][vname] = {"r2_mean": r2_mean, "r2_std": r2_std}
                print(f"  nf={nf:2d} D={D:3d} goal={g:10s} {vname:22s} R2={r2_mean:+.3f} +- {r2_std:.3f}", flush=True)

    with open(OUT / "stage0_results.json", "w") as f:
        json.dump({"results": results, "n_fourier_grid": N_FOURIER_GRID,
                    "goal_names": GOAL_NAMES, "value_names": VALUE_NAMES,
                    "per_slate_samples": PER_SLATE_SAMPLES, "seed": SEED,
                    "corpus": str(CORPUS), "provenance": git_provenance()}, f, indent=2)
    print("\nsaved stage0_results.json", flush=True)


if __name__ == "__main__":
    main()
