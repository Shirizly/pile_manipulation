"""Persist fitted objects for the models actually reported in RESULTS.md:
stage2's diff-ridge (6 fixed goals), stage3's diff-ridge (500 goals, fit on
ALL goals), and stage4's linear-ridge + polynomial-ridge (held-out-goal
split). Each saved with its feature spec / config / provenance.
Sonnet 5 note: MLP from stage4 intentionally NOT refit/persisted here --
budget; its R^2 is recorded in stage4_results.json and RESULTS.md instead
(reported, not reloadable).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sklearn.linear_model import RidgeCV                              # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline                            # noqa: E402
from utils import git_provenance                                      # noqa: E402

OUT = Path(__file__).resolve().parents[1]
PROV = git_provenance()


def ridge_fit(Xtr, ytr, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
    Xs = (Xtr - mu) / sd
    f = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(f)
    b = Xs.T @ ytr
    w = np.linalg.solve(A, b)
    return {"mu": mu, "sd": sd, "w": w, "b0": ytr.mean(), "lam": lam}


LAM_GRID = [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]


def cv_lambda(X, y, lam_grid, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    folds = np.array_split(rng.permutation(n), 3)
    best_lam, best_score = lam_grid[0], -np.inf
    for lam in lam_grid:
        scores = []
        for k in range(3):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(3) if j != k])
            m = ridge_fit(X[tr], y[tr], lam)
            Xs = (X[te] - m["mu"]) / m["sd"]
            pred = m["b0"] + Xs @ m["w"]
            ss_res = np.sum((y[te] - pred) ** 2); ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
            scores.append(1 - ss_res / max(ss_tot, 1e-12))
        s = np.mean(scores)
        if s > best_score:
            best_score, best_lam = s, lam
    return best_lam


VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
fitted = {}

# --- stage 2: 6 fixed goals, diff features, fit on ALL (state,goal) pairs ---
d2 = torch.load(OUT / "stage2_features_and_targets.pt", weights_only=False)
phi_state2 = d2["phi_state"]; phi_goal2 = d2["phi_goal"]; values2 = d2["values"]
S2, D2 = phi_state2.shape; G2 = phi_goal2.shape[0]
diff2 = np.broadcast_to(phi_state2[:, None, :] - phi_goal2[None, :, :], (S2, G2, D2)).reshape(-1, D2)
for vname in VALUE_NAMES:
    y = values2[vname].reshape(-1)
    lam = cv_lambda(diff2, y, LAM_GRID)
    model = ridge_fit(diff2, y, lam)
    fitted[("stage2_configs6", vname, "diff")] = {
        **model, "feature_mode": "diff", "n_fourier": d2["n_fourier"],
        "goal_source": "configuration (Baselines/common/goal_configs.mask_to_configuration, "
                        "K=20 samples averaged)", "goal_names": d2["goal_names"],
        "provenance": PROV}
print("stage2 models fit.", flush=True)

# --- stage 3: 500 goals, diff features, fit on ALL (state,goal) pairs ---
d3 = torch.load(OUT / "stage3_features_and_targets.pt", weights_only=False)
phi_state3 = d3["phi_state"]; phi_goal3 = d3["phi_goal"]; values3 = d3["values"]
S3, D3 = phi_state3.shape; G3 = phi_goal3.shape[0]
diff3 = np.broadcast_to(phi_state3[:, None, :] - phi_goal3[None, :, :], (S3, G3, D3)).reshape(-1, D3)
for vname in VALUE_NAMES:
    y = values3[vname].reshape(-1)
    lam = cv_lambda(diff3, y, LAM_GRID)
    model = ridge_fit(diff3, y, lam)
    fitted[("stage3_configs500", vname, "diff")] = {
        **model, "feature_mode": "diff", "n_fourier": d3["n_fourier"],
        "goal_source": "configuration, K=3 samples averaged, 500 goals "
                        "(experiments/temp/goal-states/dataset.pt)",
        "provenance": PROV}
print("stage3 models fit.", flush=True)

with open(OUT / "fitted_models.pkl", "wb") as f:
    pickle.dump(fitted, f)
print(f"saved fitted_models.pkl ({len(fitted)} models: stage2 x3, stage3 x3)", flush=True)

# --- stage 4: linear + poly2 ridge pipelines, held-out-goal split (mass_in_region only,
# the value fn where diff underperforms goal_only at state-split -- most decision-relevant) ---
D = phi_state3.shape[1]
rng = np.random.default_rng(777)
gperm = rng.permutation(G3)
test_g, train_g = gperm[:100], gperm[100:]
diff_full = np.broadcast_to(phi_state3[:, None, :] - phi_goal3[None, :, :], (S3, G3, D))
for vname in ["mass_in_region"]:
    y = values3[vname]
    Xtr = diff_full[:, train_g, :].reshape(-1, D)
    ytr = y[:, train_g].reshape(-1)
    sub = np.random.default_rng(0).choice(len(ytr), size=min(60000, len(ytr)), replace=False)
    Xtr_s, ytr_s = Xtr[sub], ytr[sub]
    lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 20))).fit(Xtr_s, ytr_s)
    joblib.dump({"pipeline": lin, "feature_mode": "diff", "n_fourier": d3["n_fourier"],
                 "train_goals_idx": train_g, "test_goals_idx": test_g,
                 "value_fn": vname, "provenance": PROV},
                OUT / f"stage4_linear_{vname}.joblib")
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
                          StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 12))).fit(Xtr_s, ytr_s)
    joblib.dump({"pipeline": poly, "feature_mode": "diff_poly2", "n_fourier": d3["n_fourier"],
                 "train_goals_idx": train_g, "test_goals_idx": test_g,
                 "value_fn": vname, "provenance": PROV},
                OUT / f"stage4_poly2_{vname}.joblib")
    print(f"saved stage4 linear+poly2 models for {vname}", flush=True)

print("done.", flush=True)
