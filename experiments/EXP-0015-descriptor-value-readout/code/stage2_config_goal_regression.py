"""Stage 2 — repeat the Stage-1 (desc-value-regression) test, but phi(goal)
now comes from a generated LEGAL configuration (Baselines/common/goal_configs)
instead of the solid mask. Same 6 goal shapes, 3 value functions, mean/state/
goal baselines, held-out-state and held-out-goal splits, ridge+CV protocol --
directly comparable to experiments/temp/desc-value-regression/RESULTS.md.

Value targets are computed from the goal REGION (mask / distance field), NOT
from the sampled configuration -- the configuration only supplies phi(goal).
phi(goal) itself is the descriptor of the AVERAGE occupancy over K independent
configuration samples (stabilises against single-sample placement noise);
per-sample spread is reported separately as the stability diagnostic the brief
asks for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from control_utility_test import lyapunov                        # noqa: E402
from dmdc_baseline import occupancy_descriptors, descriptor_slices  # noqa: E402
from transforms.functional import particles_to_occupancy          # noqa: E402
from Baselines.common.goals import (                              # noqa: E402
    quadrant_mask, letter_mask, dist_field_from_mask,
    mass_in_region, signed_mass_in_region,
)
from Baselines.common.goal_configs import (                       # noqa: E402
    mask_to_configuration, assert_no_penetration, DEFAULT_BOUNDS, CUBE_SIZE,
)
from Genesis.binned_slate_dataset import BinnedSlateCorpus        # noqa: E402
from utils import git_provenance                                  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
CORPUS = REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"

BOUNDS = DEFAULT_BOUNDS
GRID = 64
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
DEVICE = "cpu"
N_FOURIER = 8
PER_SLATE_SAMPLES = 200
SEED = 0
K_CONFIG_SAMPLES = 20  # independent configuration samples per goal, for phi(goal) + stability

GOAL_NAMES = ["quadrant0", "quadrant1", "quadrant2", "quadrant3", "letter_T", "letter_O"]
VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
LAM_GRID = [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]
rng_cv = np.random.default_rng(0)


def build_masks():
    return {
        "quadrant0": quadrant_mask(GRID, GRID, 0), "quadrant1": quadrant_mask(GRID, GRID, 1),
        "quadrant2": quadrant_mask(GRID, GRID, 2), "quadrant3": quadrant_mask(GRID, GRID, 3),
        "letter_T": letter_mask("T", GRID, GRID), "letter_O": letter_mask("O", GRID, GRID),
    }


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
    n = Xtr.shape[0]
    if Xtr.shape[1] == 0:
        pred = np.full(len(yte), ytr.mean())
        return {"lam": None, "r2": r2(yte, pred)}
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


def split_state(slate_of_state, n_slates, n_test_slates=4, seed=0):
    r = np.random.default_rng(seed)
    slates = np.arange(n_slates); r.shuffle(slates)
    test_slates = set(slates[:n_test_slates].tolist())
    is_test = np.array([s in test_slates for s in slate_of_state])
    return ~is_test, is_test


def make_features(mode, phi_state, phi_goal):
    S, D = phi_state.shape; G = phi_goal.shape[0]
    ps = phi_state[:, None, :]; pg = phi_goal[None, :, :]
    diff = np.broadcast_to(ps - pg, (S, G, D))
    if mode == "mean_only":
        return np.zeros((S, G, 0))
    if mode == "state_only":
        return np.broadcast_to(ps, (S, G, D))
    if mode == "goal_only":
        return np.broadcast_to(pg, (S, G, D))
    if mode == "diff":
        return diff
    raise ValueError(mode)


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

    print("rasterising state occupancies + descriptors...", flush=True)
    occ_state = chunked(lambda a, b: occ_of(states_[a:b]), len(idx))
    phi_state = occupancy_descriptors(occ_state, n_fourier=N_FOURIER).numpy().astype(np.float64)

    masks = build_masks()
    dist_fields = {g: torch.from_numpy(dist_field_from_mask(masks[g])) for g in GOAL_NAMES}
    mask_t = {g: torch.from_numpy(masks[g].astype(np.float32)) for g in GOAL_NAMES}

    print(f"generating {K_CONFIG_SAMPLES} configuration samples per goal...", flush=True)
    phi_goal_samples = {}   # goal -> (K, D) descriptor of EACH sample (stability diagnostic)
    phi_goal_avg = {}       # goal -> (D,) descriptor of the AVERAGE occupancy across samples
    gen_methods = {}
    avg_occ_store = {}
    for g in GOAL_NAMES:
        occs = []
        methods = set()
        for k in range(K_CONFIG_SAMPLES):
            res = mask_to_configuration(masks[g], n_objects=20, seed=(hash(g) % 10000) * 1000 + k)
            assert_no_penetration(res.poses[:, :2])
            occ_k = occ_of(torch.from_numpy(res.poses[None, :, :3]).float())[0]
            occs.append(occ_k)
            methods.add(res.method)
        occs = torch.stack(occs)  # (K,H,W)
        phi_each = occupancy_descriptors(occs, n_fourier=N_FOURIER).numpy().astype(np.float64)
        phi_goal_samples[g] = phi_each
        avg_occ = occs.mean(dim=0, keepdim=True)  # (1,H,W)
        avg_occ_store[g] = avg_occ
        phi_goal_avg[g] = occupancy_descriptors(avg_occ, n_fourier=N_FOURIER).numpy().astype(np.float64)[0]
        gen_methods[g] = sorted(methods)
        print(f"  {g}: methods={gen_methods[g]}", flush=True)

    # Stability diagnostic: per-sample spread vs between-goal spread
    stability = {}
    all_avg = np.stack([phi_goal_avg[g] for g in GOAL_NAMES])  # (G,D)
    between_goal_std = all_avg.std(axis=0)  # (D,)
    for g in GOAL_NAMES:
        within_std = phi_goal_samples[g].std(axis=0)  # (D,)
        # pooled scalar ratio (L2 norm of within-std / L2 norm of between-goal-std)
        ratio = float(np.linalg.norm(within_std) / max(np.linalg.norm(between_goal_std), 1e-9))
        stability[g] = {"within_sample_std_l2": float(np.linalg.norm(within_std)),
                         "between_goal_std_l2": float(np.linalg.norm(between_goal_std)),
                         "ratio_within_over_between": ratio}
        print(f"  stability {g}: within/between L2 ratio = {ratio:.4f}", flush=True)

    phi_goal = np.stack([phi_goal_avg[g] for g in GOAL_NAMES])  # (G,D) -- used in regression

    # value targets from the MASK (not the sampled configuration)
    values = {"lyapunov": torch.zeros(len(idx), len(GOAL_NAMES)),
              "mass_in_region": torch.zeros(len(idx), len(GOAL_NAMES)),
              "signed_mass_in_region": torch.zeros(len(idx), len(GOAL_NAMES))}
    for gi, g in enumerate(GOAL_NAMES):
        values["lyapunov"][:, gi] = lyapunov(occ_state, dist_fields[g])
        values["mass_in_region"][:, gi] = mass_in_region(occ_state, mask_t[g])
        values["signed_mass_in_region"][:, gi] = signed_mass_in_region(occ_state, mask_t[g])
    values = {k: v.numpy().astype(np.float64) for k, v in values.items()}

    print("\n=== regressions ===", flush=True)
    results = {"state_split": {}, "goal_split": {}, "stability": stability, "gen_methods": gen_methods}
    modes = ["mean_only", "state_only", "goal_only", "diff"]
    G = len(GOAL_NAMES)
    for vname in VALUE_NAMES:
        results["state_split"][vname] = {}
        results["goal_split"][vname] = {}
        y_full = values[vname]
        for mode in modes:
            feat = make_features(mode, phi_state, phi_goal)
            F = feat.shape[-1]
            # state split, 3 repeats
            r2s = []
            for rep in range(3):
                tr_mask, te_mask = split_state(slate_of_state, n_slates, seed=rep)
                ytr = y_full[tr_mask].reshape(-1); yte = y_full[te_mask].reshape(-1)
                Xtr = feat[tr_mask].reshape(len(ytr), F) if F > 0 else np.zeros((len(ytr), 0))
                Xte = feat[te_mask].reshape(len(yte), F) if F > 0 else np.zeros((len(yte), 0))
                res = fit_eval_cv(Xtr, ytr, Xte, yte, LAM_GRID)
                r2s.append(res["r2"])
            results["state_split"][vname][mode] = {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
            # goal split, leave-one-out
            r2s_g = []
            for gi in range(G):
                train_g = [j for j in range(G) if j != gi]
                ytr = y_full[:, train_g].reshape(-1); yte = y_full[:, gi]
                Xtr = feat[:, train_g, :].reshape(len(ytr), F) if F > 0 else np.zeros((len(ytr), 0))
                Xte = feat[:, gi, :] if F > 0 else np.zeros((len(yte), 0))
                res = fit_eval_cv(Xtr, ytr, Xte, yte, LAM_GRID)
                r2s_g.append(res["r2"])
            results["goal_split"][vname][mode] = {"r2_mean": float(np.mean(r2s_g)), "r2_std": float(np.std(r2s_g)),
                                                    "per_goal": {GOAL_NAMES[gi]: r2s_g[gi] for gi in range(G)}}
            print(f"{vname:22s} {mode:10s} state-split R2={results['state_split'][vname][mode]['r2_mean']:+.3f}"
                  f"+-{results['state_split'][vname][mode]['r2_std']:.3f}  "
                  f"goal-split R2={results['goal_split'][vname][mode]['r2_mean']:+.3f}"
                  f"+-{results['goal_split'][vname][mode]['r2_std']:.3f}", flush=True)

    with open(OUT / "stage2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({"phi_state": phi_state, "phi_goal": phi_goal, "phi_goal_samples": phi_goal_samples,
                "slate_of_state": slate_of_state, "goal_names": GOAL_NAMES,
                "descriptor_slices": descriptor_slices(N_FOURIER), "values": values,
                "n_fourier": N_FOURIER, "k_config_samples": K_CONFIG_SAMPLES,
                "corpus": str(CORPUS), "provenance": git_provenance()},
               OUT / "stage2_features_and_targets.pt")
    print("\nsaved stage2_results.json, stage2_features_and_targets.pt", flush=True)


if __name__ == "__main__":
    main()
