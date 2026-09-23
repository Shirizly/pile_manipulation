"""Stage 3 (EXP-0018) -- fit value readouts on the diverse-state cache.

Three split families, all built over GROUPS, never rows:
  goal    -- entire goal shapes withheld (80/20 over the 2000-goal library)
  state   -- unseen states; groups (source file / multistep slate) withheld, goals shared
  corpus  -- leave-one-source-corpus-out; goals shared

Inner ridge-lambda CV is GroupKFold (`cv_group_by`), grouped by state-group for the
state/corpus splits and by goal id for the goal split. Baselines mean_only /
state_only / goal_only are computed by ValueReadout at the same capacity on every fit;
the reported baseline is the max over capacities (each baseline at its OWN best capacity).

Writes incrementally to results/metrics.json and persists every fitted readout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0017-value-readout-instrument/code"))

from value_readout import ValueReadout                                # noqa: E402
from utils import git_provenance                                      # noqa: E402

CACHE = REPO / "experiments/temp/exp0018-value-readout/cache.pt"
OUTDIR = REPO / "experiments/temp/exp0018-value-readout"
RES = REPO / "experiments/EXP-0018-value-readout-diverse-states/results"
RES.mkdir(parents=True, exist_ok=True)

VALUES = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
N_FIT_STATES = 6000        # stratified subsample of the 23520 for tractable fits
GOAL_TEST_FRAC = 0.2
N_GOAL_REPS = 3
N_STATE_REPS = 2
FEATURE_MODE = "diff"      # 87-dim state descriptor minus 87-dim goal descriptor

MLP_KW = dict(capacity="mlp", feature_mode=FEATURE_MODE, hidden=(128, 128),
              mlp_max_iter=120, max_train_rows=20000, max_eval_rows=15000)
LIN_KW = dict(capacity="linear", feature_mode=FEATURE_MODE,
              max_train_rows=30000, max_eval_rows=20000)
POLY_KW = dict(capacity="poly2", feature_mode=FEATURE_MODE,
               poly_alphas=tuple(np.logspace(1, 6, 4)),
               max_train_rows=6000, max_eval_rows=8000)

results = []
RESFILE = RES / "metrics.json"


def flush():
    json.dump({"metrics_key": "value_readout_r2", "cells": results,
               "provenance": git_provenance(),
               "n_fit_states": N_FIT_STATES, "feature_mode": FEATURE_MODE},
              open(RESFILE, "w"), indent=1, default=str)


def main():
    c = torch.load(CACHE, weights_only=False)
    phi_s = np.asarray(c["phi_state"], dtype=np.float64)
    phi_g = np.asarray(c["phi_goal"], dtype=np.float64)
    corpus = np.asarray(c["corpus"])
    group = np.asarray(c["group"])
    G = phi_g.shape[0]

    rng = np.random.default_rng(0)
    # stratified subsample of states, proportional per corpus, groups kept intact
    sel = []
    for cname in np.unique(corpus):
        idx = np.nonzero(corpus == cname)[0]
        k = int(round(N_FIT_STATES * len(idx) / len(corpus)))
        sel.append(rng.choice(idx, min(k, len(idx)), replace=False))
    sub = np.sort(np.concatenate(sel))
    np.save(OUTDIR / "fit_state_subsample_idx.npy", sub)
    phi_s_f = phi_s[sub]
    corpus_f = corpus[sub]
    group_f = group[sub]
    _, gid = np.unique(group_f, return_inverse=True)
    S = len(sub)
    print(f"fit states S={S}; corpora={dict(zip(*np.unique(corpus_f, return_counts=True)))}; "
          f"groups={len(np.unique(gid))}", flush=True)

    yall = {v: np.asarray(c["values"][v], dtype=np.float64)[sub] for v in VALUES}
    all_states = np.arange(S)
    all_goals = np.arange(G)

    splits = []
    # --- held-out GOAL -----------------------------------------------------
    for rep in range(N_GOAL_REPS):
        r = np.random.default_rng(100 + rep)
        perm = r.permutation(G)
        nte = int(GOAL_TEST_FRAC * G)
        splits.append(("goal", f"rep{rep}",
                       dict(train_states=all_states, test_states=all_states,
                            train_goals=perm[nte:], test_goals=perm[:nte]), "goal"))
    # --- held-out SOURCE CORPUS -------------------------------------------
    for cname in sorted(np.unique(corpus_f)):
        te = np.nonzero(corpus_f == cname)[0]
        tr = np.nonzero(corpus_f != cname)[0]
        splits.append(("corpus", cname,
                       dict(train_states=tr, test_states=te,
                            train_goals=all_goals, test_goals=all_goals), "state"))
    # --- held-out STATE (grouped) -----------------------------------------
    for rep in range(N_STATE_REPS):
        r = np.random.default_rng(200 + rep)
        gids = np.unique(gid)
        r.shuffle(gids)
        test_g = set(gids[:max(1, int(0.2 * len(gids)))].tolist())
        is_te = np.array([g in test_g for g in gid])
        splits.append(("state", f"rep{rep}",
                       dict(train_states=np.nonzero(~is_te)[0],
                            test_states=np.nonzero(is_te)[0],
                            train_goals=all_goals, test_goals=all_goals), "state"))

    # cell order = priority order; a truncated run still has the cells that matter
    plan = []
    for fam, name, sp, cvby in splits:
        caps = ["linear", "mlp"]
        if fam == "corpus":
            for v in VALUES:
                plan.append((fam, name, sp, cvby, "linear", v))
            plan.append((fam, name, sp, cvby, "mlp", "lyapunov"))
        else:
            for cap in caps:
                for v in VALUES:
                    plan.append((fam, name, sp, cvby, cap, v))
    # poly2 capacity reference, last
    plan.append(("goal", "rep0", splits[0][2], "goal", "poly2", "lyapunov"))

    t0 = time.time()
    for i, (fam, name, sp, cvby, cap, vname) in enumerate(plan):
        kw = {"linear": LIN_KW, "mlp": MLP_KW, "poly2": POLY_KW}[cap]
        ro = ValueReadout(**kw, seed=0,
                          notes=f"EXP-0018 {fam}/{name} {cap} {vname}",
                          provenance={"cache": str(CACHE), "split_family": fam,
                                      "split_name": name, "value": vname})
        ro.cfg.capacity = cap
        m = ro.fit(phi_s_f, phi_g, yall[vname], sp,
                   state_groups=gid, goal_groups=np.arange(G), cv_group_by=cvby)
        m.update({"split_family": fam, "split_name": name, "value": vname,
                  "capacity": cap, "n_train_states": int(len(sp["train_states"])),
                  "n_test_states": int(len(sp["test_states"])),
                  "n_train_goals": int(len(sp["train_goals"])),
                  "n_test_goals": int(len(sp["test_goals"]))})
        results.append(m)
        path = OUTDIR / f"readout__{fam}__{name}__{cap}__{vname}.joblib"
        ro.save(path)
        flush()
        print(f"[{i+1}/{len(plan)} {time.time()-t0:6.0f}s] {fam:6s}/{name:16s} {cap:6s} "
              f"{vname:22s} value_readout_r2={m['r2']:+.4f} "
              f"goal_only={m['baseline_goal_only']:+.4f} "
              f"state_only={m['baseline_state_only']:+.4f} "
              f"incr={m['increment_over_goal_only']:+.4f} alpha={m.get('alpha_selected')}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
