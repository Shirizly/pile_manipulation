"""Stage 4 — capacity escalation, held-out-GOAL only (the meaningful split,
per stage 3). Linear ridge (baseline, reproduced) -> degree-2 polynomial
ridge -> small MLP, on the same 500-goal diff features. One goal-split
(400/100) repeat per level (budget-bounded) rather than stage 3's 3 repeats.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from sklearn.linear_model import RidgeCV                    # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.neural_network import MLPRegressor              # noqa: E402
from sklearn.metrics import r2_score                          # noqa: E402

OUT = Path(__file__).resolve().parents[1]
DATA = torch.load(OUT / "stage3_features_and_targets.pt", weights_only=False)
phi_state = DATA["phi_state"]  # (S,D)
phi_goal = DATA["phi_goal"]    # (G,D)
values = DATA["values"]        # dict vname -> (S,G)
S, D = phi_state.shape
G = phi_goal.shape[0]
VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
N_TEST_GOALS = 100

rng = np.random.default_rng(777)
gperm = rng.permutation(G)
test_g, train_g = gperm[:N_TEST_GOALS], gperm[N_TEST_GOALS:]

ps = phi_state[:, None, :]; pg = phi_goal[None, :, :]
diff_full = np.broadcast_to(ps - pg, (S, G, D))


def build(train_g, test_g, y):
    Xtr = diff_full[:, train_g, :].reshape(-1, D)
    Xte = diff_full[:, test_g, :].reshape(-1, D)
    ytr = y[:, train_g].reshape(-1)
    yte = y[:, test_g].reshape(-1)
    return Xtr, ytr, Xte, yte


results = {}
for vname in VALUE_NAMES:
    y = values[vname]
    Xtr, ytr, Xte, yte = build(train_g, test_g, y)
    results[vname] = {}
    print(f"\n=== {vname} (held-out-GOAL, {len(train_g)} train / {len(test_g)} test goals) ===", flush=True)

    # subsample rows for speed (poly/MLP on 400*1000=400k rows is slow) --
    # 60k train rows is plenty for D<=87*88/2 features and an MLP this size.
    sub = np.random.default_rng(0).choice(len(ytr), size=min(60000, len(ytr)), replace=False)
    Xtr_s, ytr_s = Xtr[sub], ytr[sub]

    t0 = time.time()
    lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 20)))
    lin.fit(Xtr_s, ytr_s)
    r2_lin = r2_score(yte, lin.predict(Xte))
    print(f"  linear ridge      R2={r2_lin:+.3f}  ({time.time()-t0:.1f}s)", flush=True)
    results[vname]["linear"] = r2_lin

    t0 = time.time()
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, interaction_only=False,
                          include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 12)))
    try:
        poly.fit(Xtr_s, ytr_s)
        r2_poly = r2_score(yte, poly.predict(Xte))
    except MemoryError:
        r2_poly = float("nan")
    print(f"  degree-2 poly ridge R2={r2_poly:+.3f}  ({time.time()-t0:.1f}s)", flush=True)
    results[vname]["poly2"] = r2_poly

    t0 = time.time()
    mlp = make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                                      alpha=1e-3, max_iter=300, early_stopping=True,
                                      n_iter_no_change=10, random_state=0))
    mlp.fit(Xtr_s, ytr_s)
    r2_mlp = r2_score(yte, mlp.predict(Xte))
    print(f"  MLP(128,128)      R2={r2_mlp:+.3f}  ({time.time()-t0:.1f}s)", flush=True)
    results[vname]["mlp"] = r2_mlp

with open(OUT / "stage4_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved stage4_results.json", flush=True)
