"""Stage 3b — repeat the config-goal regression (stage2) with the ~500-goal
dataset from stage3a, making held-out-GOAL a meaningful test (random
train/test splits over goals, repeated, rather than 6-fold leave-one-out).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from control_utility_test import lyapunov                          # noqa: E402
from dmdc_baseline import occupancy_descriptors                     # noqa: E402
from transforms.functional import particles_to_occupancy            # noqa: E402
from Baselines.common.goals import (                                # noqa: E402
    dist_field_from_mask, mass_in_region, signed_mass_in_region,
)
from Baselines.common.goal_configs import DEFAULT_BOUNDS, CUBE_SIZE  # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus          # noqa: E402
from utils import git_provenance                                    # noqa: E402

OUT = Path(__file__).resolve().parents[1]
CORPUS = REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"
GOALS_DATASET = REPO / "experiments/temp/goal-states/dataset.pt"

BOUNDS = DEFAULT_BOUNDS
GRID = 64
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
DEVICE = "cpu"
N_FOURIER = 8
PER_SLATE_SAMPLES = 50   # reduced from stage0/2's 200 to keep S*G tractable
SEED = 0
VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
LAM_GRID = [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]
N_GOAL_SPLIT_REPEATS = 3
N_TEST_GOALS = 100  # out of 500
rng_cv = np.random.default_rng(0)


def occ_of(states):
    return particles_to_occupancy(states[..., :3].to(DEVICE), BOUNDS, (GRID, GRID), footprint_radius=RADIUS)


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
    if Xtr.shape[1] == 0:
        pred = np.full(len(yte), ytr.mean())
        return {"lam": None, "r2": r2(yte, pred)}
    n = Xtr.shape[0]
    folds = np.array_split(rng_cv.permutation(n), 3)
    best_lam, best_score = lam_grid[0], -np.inf
    for lam in lam_grid:
        scores = []
        for k in range(3):
            te_idx = folds[k]; tr_idx = np.concatenate([folds[j] for j in range(3) if j != k])
            m = ridge_fit(Xtr[tr_idx], ytr[tr_idx], lam)
            scores.append(r2(ytr[te_idx], ridge_predict(m, Xtr[te_idx])))
        s = np.mean(scores)
        if s > best_score:
            best_score, best_lam = s, lam
    model = ridge_fit(Xtr, ytr, best_lam)
    pred = ridge_predict(model, Xte)
    return {"lam": best_lam, "r2": r2(yte, pred), "model": model}


def make_features(mode, phi_state, phi_goal):
    S, D = phi_state.shape; G = phi_goal.shape[0]
    ps = phi_state[:, None, :]; pg = phi_goal[None, :, :]
    if mode == "mean_only":
        return np.zeros((S, G, 0))
    if mode == "state_only":
        return np.broadcast_to(ps, (S, G, D))
    if mode == "goal_only":
        return np.broadcast_to(pg, (S, G, D))
    if mode == "diff":
        return np.broadcast_to(ps - pg, (S, G, D))
    raise ValueError(mode)


def split_state(slate_of_state, n_slates, n_test_slates=4, seed=0):
    r = np.random.default_rng(seed)
    slates = np.arange(n_slates); r.shuffle(slates)
    test_slates = set(slates[:n_test_slates].tolist())
    is_test = np.array([s in test_slates for s in slate_of_state])
    return ~is_test, is_test


def main():
    print("loading corpus + goal dataset...", flush=True)
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
    S = len(idx)
    print(f"S={S} states", flush=True)

    occ_state = chunked(lambda a, b: occ_of(states_[a:b]), S)
    phi_state = occupancy_descriptors(occ_state, n_fourier=N_FOURIER).numpy().astype(np.float64)

    gd = torch.load(GOALS_DATASET, weights_only=False)
    masks = gd["masks"].numpy()          # (G,H,W) bool
    configs = gd["configs"]              # (G,K,20,7)
    metas = gd["metadata"]
    G, K = configs.shape[0], configs.shape[1]
    print(f"G={G} goals, K={K} config samples each", flush=True)

    print("rasterising goal configurations -> phi_goal (avg occupancy over K samples)...", flush=True)
    phi_goal = np.zeros((G, phi_state.shape[1]), dtype=np.float64)
    BATCH = 50
    for lo in range(0, G, BATCH):
        hi = min(lo + BATCH, G)
        poses = configs[lo:hi].reshape(-1, 20, 7)  # (batch*K, 20, 7)
        occ_k = occ_of(poses.float())               # (batch*K, H, W)
        occ_k = occ_k.reshape(hi - lo, K, GRID, GRID).mean(dim=1)  # (batch,H,W)
        phi_goal[lo:hi] = occupancy_descriptors(occ_k, n_fourier=N_FOURIER).numpy().astype(np.float64)
    print("done", flush=True)

    print("computing value targets from goal MASKS (not sampled configs)...", flush=True)
    values = {vname: np.zeros((S, G), dtype=np.float64) for vname in VALUE_NAMES}
    for gi in range(G):
        m = masks[gi]
        d = torch.from_numpy(dist_field_from_mask(m))
        mt = torch.from_numpy(m.astype(np.float32))
        values["lyapunov"][:, gi] = lyapunov(occ_state, d).numpy()
        values["mass_in_region"][:, gi] = mass_in_region(occ_state, mt).numpy()
        values["signed_mass_in_region"][:, gi] = signed_mass_in_region(occ_state, mt).numpy()
    print("done", flush=True)

    print("\n=== state-split (500 goals fixed) ===", flush=True)
    results = {"state_split": {}, "goal_split": {}, "n_states": S, "n_goals": G}
    modes = ["mean_only", "state_only", "goal_only", "diff"]
    for vname in VALUE_NAMES:
        results["state_split"][vname] = {}
        y_full = values[vname]
        for mode in modes:
            feat = make_features(mode, phi_state, phi_goal)
            F = feat.shape[-1]
            r2s = []
            for rep in range(3):
                tr_mask, te_mask = split_state(slate_of_state, n_slates, seed=rep)
                ytr = y_full[tr_mask].reshape(-1); yte = y_full[te_mask].reshape(-1)
                Xtr = feat[tr_mask].reshape(len(ytr), F) if F > 0 else np.zeros((len(ytr), 0))
                Xte = feat[te_mask].reshape(len(yte), F) if F > 0 else np.zeros((len(yte), 0))
                res = fit_eval_cv(Xtr, ytr, Xte, yte, LAM_GRID)
                r2s.append(res["r2"])
            results["state_split"][vname][mode] = {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
            print(f"  {vname:22s} {mode:10s} R2={np.mean(r2s):+.3f}+-{np.std(r2s):.3f}", flush=True)

    print("\n=== held-out-GOAL split (random 400/100, %d repeats) ===" % N_GOAL_SPLIT_REPEATS, flush=True)
    for vname in VALUE_NAMES:
        results["goal_split"][vname] = {}
        y_full = values[vname]
        for mode in modes:
            feat = make_features(mode, phi_state, phi_goal)
            F = feat.shape[-1]
            r2s = []
            for rep in range(N_GOAL_SPLIT_REPEATS):
                r = np.random.default_rng(100 + rep)
                gperm = r.permutation(G)
                test_g = gperm[:N_TEST_GOALS]; train_g = gperm[N_TEST_GOALS:]
                ytr = y_full[:, train_g].reshape(-1); yte = y_full[:, test_g].reshape(-1)
                Xtr = feat[:, train_g, :].reshape(len(ytr), F) if F > 0 else np.zeros((len(ytr), 0))
                Xte = feat[:, test_g, :].reshape(len(yte), F) if F > 0 else np.zeros((len(yte), 0))
                res = fit_eval_cv(Xtr, ytr, Xte, yte, LAM_GRID)
                r2s.append(res["r2"])
            results["goal_split"][vname][mode] = {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
            print(f"  {vname:22s} {mode:10s} R2={np.mean(r2s):+.3f}+-{np.std(r2s):.3f}", flush=True)

    with open(OUT / "stage3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({"phi_state": phi_state, "phi_goal": phi_goal, "slate_of_state": slate_of_state,
                "values": values, "n_fourier": N_FOURIER, "per_slate_samples": PER_SLATE_SAMPLES,
                "goals_dataset": str(GOALS_DATASET), "corpus": str(CORPUS),
                "provenance": git_provenance()}, OUT / "stage3_features_and_targets.pt")
    print("\nsaved stage3_results.json, stage3_features_and_targets.pt", flush=True)


if __name__ == "__main__":
    main()
