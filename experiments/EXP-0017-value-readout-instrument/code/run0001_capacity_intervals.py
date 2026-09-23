"""RUN-0001 part A -- capacity x value-function x REPEATED goal splits, with the
three mandatory baselines at each capacity, and the MLP PERSISTED.

Closes EXP-0015 defects (1) persist the MLP and (2) put an interval on the
capacity ordering (EXP-0015's stage4 was a single goal split).
Also re-measures the held-out-STATE cells that went negative at 500 goals.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from value_readout import ValueReadout          # noqa: E402
from utils import git_provenance                # noqa: E402

DATA = REPO / "experiments/temp/desc-value-readout/stage3_features_and_targets.pt"
ART = EXP / "artifacts/RUN-0001"; ART.mkdir(parents=True, exist_ok=True)
RES = EXP / "results"; RES.mkdir(parents=True, exist_ok=True)
MODELS = REPO / "experiments/temp/exp0017-value-readout"; MODELS.mkdir(parents=True, exist_ok=True)

VALUE_NAMES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
N_TEST_GOALS = 100
GOAL_SPLIT_SEEDS = [100, 101, 102, 103]        # >=3, as briefed
STATE_SPLIT_SEEDS = [0, 1, 2]
N_TEST_SLATES = 4
MAX_TRAIN, MAX_EVAL = 25000, 20000
PROV = git_provenance()

d = torch.load(DATA, weights_only=False)
Xs, Xg, values = d["phi_state"], d["phi_goal"], d["values"]
S, G = len(Xs), len(Xg)
slate_of_state = np.asarray(d["slate_of_state"])
slates = np.unique(slate_of_state)
print(f"S={S} states, G={G} goals, D={Xs.shape[1]}, n_fourier={d['n_fourier']}", flush=True)

def goal_split(seed):
    p = np.random.default_rng(seed).permutation(G)
    return dict(train_states=np.arange(S), test_states=np.arange(S),
                train_goals=p[N_TEST_GOALS:], test_goals=p[:N_TEST_GOALS])

def state_split(seed):
    p = np.random.default_rng(seed).permutation(slates)
    te = set(p[:N_TEST_SLATES].tolist())
    is_te = np.array([s in te for s in slate_of_state])
    return dict(train_states=np.nonzero(~is_te)[0], test_states=np.nonzero(is_te)[0],
                train_goals=np.arange(G), test_goals=np.arange(G))

# which (capacity -> modes, n_splits) cells to run, budget-bounded; declared up front
PLAN_GOAL = [
    ("linear", ["diff", "concat", "goal_only", "state_only", "mean_only"], 4),
    ("mlp",    ["diff", "goal_only"], 3),
    ("poly2",  ["diff"], 3),
    ("poly2",  ["goal_only"], 1),
]
PLAN_STATE = [
    ("linear", ["diff", "goal_only", "state_only", "mean_only"], 3),
    ("mlp",    ["diff", "goal_only"], 1),
]

rows = []
def sweep(plan, split_fn, seeds, split_kind, persist_capacity=None):
    for cap, modes, nsp in plan:
        for mode in modes:
            for si, seed in enumerate(seeds[:nsp]):
                for vname in VALUE_NAMES:
                    t0 = time.time()
                    ro = ValueReadout(capacity=cap, feature_mode=mode, seed=0,
                                      max_train_rows=MAX_TRAIN, max_eval_rows=MAX_EVAL,
                                      notes=f"EXP-0017 {split_kind} split seed={seed}",
                                      provenance=PROV)
                    m = ro.fit(Xs, Xg, values[vname], split_fn(seed), with_baselines=False)
                    rows.append({"split_kind": split_kind, "split_seed": int(seed),
                                 "capacity": cap, "mode": mode, "value_fn": vname,
                                 "r2": m["r2"], "secs": round(time.time() - t0, 1)})
                    print(f"  [{split_kind}] seed={seed} {cap:6s} {mode:10s} {vname:22s} "
                          f"R2={m['r2']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
                    if (persist_capacity and cap == persist_capacity and mode == "diff"
                            and si == 0 and split_kind == "goal"):
                        p = ro.save(MODELS / f"readout_{cap}_diff_{vname}_goalsplit{seed}.joblib")
                        print(f"    persisted -> {p}", flush=True)
                    json.dump(rows, open(ART / "cells.json", "w"), indent=1)

print("\n=== held-out GOAL ===", flush=True)
sweep(PLAN_GOAL, goal_split, GOAL_SPLIT_SEEDS, "goal", persist_capacity="mlp")
print("\n=== held-out STATE ===", flush=True)
sweep(PLAN_STATE, state_split, STATE_SPLIT_SEEDS, "state")

json.dump({"rows": rows, "provenance": PROV, "data": str(DATA),
           "max_train_rows": MAX_TRAIN, "max_eval_rows": MAX_EVAL,
           "goal_split_seeds": GOAL_SPLIT_SEEDS, "state_split_seeds": STATE_SPLIT_SEEDS},
          open(ART / "cells.json", "w"), indent=1)
print("\nwrote", ART / "cells.json", flush=True)
