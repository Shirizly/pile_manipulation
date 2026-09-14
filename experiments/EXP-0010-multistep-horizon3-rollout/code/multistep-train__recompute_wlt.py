"""Fix a sign bug in train_multistep.py's wins_losses_ties() and recompute
just that table (no retraining): for the COST value function (lyapunov,
higher_is_better=False) the ORIGINAL code negated vp_a/vp_b to pick the
argmax-of-negated-cost (correct, lower cost wins) but then compared the
raw (non-negated) `vt` values to decide win/loss -- inverting the sense
for lyapunov specifically. mass_in_region (higher_is_better=True) never
negates anything so its very-lopsided losses were already correct; this
script only fixes and reprints the lyapunov columns, using the operators
already saved by train_multistep.py (operator_switched_lam*.pt /
operator_global_lam*.pt) plus the same closed-form init checkpoint --
no retraining, just a corrected forward pass + comparison.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-rollout")

import numpy as np
import torch

from fit_linear_foresight import actions_to_pixels, swept_region_mask, metrics
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region
from rollout import load_dataset, DATASETS, BOUNDS, GRID, PLATE_PX, WS_MIN, WS_MAX

from train_multistep import (occ_of, px_of, apply_operator, rollout_3step,
                              make_split, LAMBDAS, DEVICE, REPO, OUT_DIR, SEED)

train_slates, test_slates = make_split(seed=SEED)

ds = {}
for tag, root in DATASETS.items():
    data, _ = load_dataset(root)
    occ0 = occ_of(data["S0"]); occ3 = occ_of(data["S3"])
    A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
    length_m = [(A[k][:, 2:4] - A[k][:, 0:2]).norm(dim=-1) for k in range(3)]
    s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
    s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]
    slate_id = data["slate_id"].numpy()
    is_test = np.isin(slate_id, list(test_slates))
    ds[tag] = dict(occ0=occ0[is_test], occ3=occ3[is_test],
                    s_px=[s[is_test] for s in s_px], e_px=[e[is_test] for e in e_px],
                    length_m=[l[is_test] for l in length_m], slate_id=slate_id[is_test])

dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
mask_corner[: GRID // 2, : GRID // 2] = True

ckpt0001 = torch.load(f"{REPO}/weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                       map_location=DEVICE, weights_only=False)
bin_edges = ckpt0001["bin_edges"].to(DEVICE)
init_ops = {"switched": [o.to(DEVICE) for o in ckpt0001["switched_ops"]],
            "global": [ckpt0001["global_op"].to(DEVICE)]}
init_edges = {"switched": bin_edges, "global": torch.zeros(2, device=DEVICE)}


def terminal_preds(ops, edges, tag):
    d = ds[tag]
    with torch.no_grad():
        preds = rollout_3step(ops, edges, d["occ0"], d["s_px"], d["e_px"], d["length_m"])
    occ3_hat = preds[2]
    v_lyap = lyapunov(occ3_hat, dw_corner).cpu().numpy()
    v_mir = mass_in_region(occ3_hat, mask_corner).cpu().numpy()
    return v_lyap, v_mir


def wlt_fixed(vp_a, vp_b, vt, higher, slate_id, uniq):
    """vp_a = trained model's predicted value, vp_b = init model's. `vt` is
    the TRUE terminal value (raw units, always). Correctly signed: for a
    COST metric (higher=False) we compare vt with an inverted sense (lower
    vt = better) rather than negating vt only on one side."""
    w = l = t = 0
    for sl in uniq:
        m = slate_id == sl
        pa, pb, tt = vp_a[m], vp_b[m], vt[m]
        ia = int(np.argmax(pa)) if higher else int(np.argmin(pa))
        ib = int(np.argmax(pb)) if higher else int(np.argmin(pb))
        if ia == ib:
            t += 1
            continue
        va, vb = tt[ia], tt[ib]
        better = (lambda x, y: x > y) if higher else (lambda x, y: x < y)
        if better(va, vb):
            w += 1
        elif better(vb, va):
            l += 1
        else:
            t += 1
    return w, l, t


out = {}
for lam in LAMBDAS:
    out[lam] = {}
    for kind in ("switched", "global"):
        fn = f"{OUT_DIR}/operator_{kind}_lam{lam}.pt"
        c = torch.load(fn, map_location=DEVICE, weights_only=False)
        ops = [o.to(DEVICE) for o in c["switched_ops"]] if kind == "switched" else [c["global_op"].to(DEVICE)]
        edges = bin_edges if kind == "switched" else torch.zeros(2, device=DEVICE)
        out[lam][kind] = {}
        for tag in DATASETS:
            v_lyap_tr, v_mir_tr = terminal_preds(ops, edges, tag)
            v_lyap_init, v_mir_init = terminal_preds(init_ops[kind], init_edges[kind], tag)
            slate_id = ds[tag]["slate_id"]
            uniq = np.unique(slate_id)
            v_true_lyap = lyapunov(ds[tag]["occ3"], dw_corner).cpu().numpy()
            v_true_mir = mass_in_region(ds[tag]["occ3"], mask_corner).cpu().numpy()
            w_l, l_l, t_l = wlt_fixed(v_lyap_tr, v_lyap_init, v_true_lyap, False, slate_id, uniq)
            w_m, l_m, t_m = wlt_fixed(v_mir_tr, v_mir_init, v_true_mir, True, slate_id, uniq)
            out[lam][kind][tag] = {
                "lyapunov": {"wins": w_l, "losses": l_l, "ties": t_l, "n": len(uniq)},
                "mass_in_region": {"wins": w_m, "losses": l_m, "ties": t_m, "n": len(uniq)},
            }
            print(f"lam={lam} {kind:9s} {tag:12s} lyap(w/l/t)={w_l}/{l_l}/{t_l}  "
                  f"mir(w/l/t)={w_m}/{l_m}/{t_m}")

with open(f"{OUT_DIR}/wlt_vs_init_fixed.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nwrote {OUT_DIR}/wlt_vs_init_fixed.json")
