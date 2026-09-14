"""Stage-2: fit switched-linear operators on state = [32x32 canonical visual
occupancy || analytic descriptors] and score them with the repo's real
image-space `accuracy` metric (experiments/METRICS.md).

Cells (see coordinator's cell-list correction, appended to this run):
  1. visual        -- 32x32 canonical occupancy only (reproduces the
                       LinearForesight res=32 baseline)
  2. hybrid14       -- visual + 14-dim D-local-minus-DFT descriptors
  2b. hybrid94      -- visual + 94-dim D-all-local descriptors (incl. DFT) --
                       PRIMARY hybrid cell per the coordinator's correction
                       (a common-target ablation in dmdc-lenbins found the
                       DFT block is a good PREDICTOR even though it's a poor
                       TARGET -- stage 1's "drop DFT" advice does not
                       transfer to "drop DFT as an input feature")
  3. desc14 / desc94 -- descriptors only, scored in IMAGE space for the
                       first time: there is no visual channel to predict, so
                       the image prediction is exactly persistence (occ0) by
                       construction; a descriptor-space secondary accuracy
                       is also reported.
  4. persistence    -- do-nothing baseline (mandatory)
  5. unswitched (global) operator for cells 1, hybrid14, hybrid94

Fit: `fit_operator` (ridge toward IDENTITY, not toward zero -- same choice
`Baselines/LinearForesight/fit_switched.py` makes and for the same reason:
the state here is occupancy-valued for the visual block, so "nothing
changed" is the right degrade-gracefully prior for an underdetermined bin;
the descriptor entries share the same operator/ridge target for simplicity,
which is defensible since several of them (band mass etc.) are also
small-and-persistence-like quantities). Ridge lam in {0.1, 1.0, 10.0} -- NOT
stage 1's {1e-4,1e-3,1e-2}, because `fit_operator` here regularises the FULL
(potentially 1000+-dim) operator toward the identity, not toward zero on a
z-scored target as stage 1's `metric.py` did; the effective scale of the
ridge term is therefore very different and the larger lam range is the one
`Baselines/LinearForesight/fit_switched.py`'s own default (ridge=1.0) sits
inside.

6 length bins, EXP-0003 scheme: equal-width over [0, max(train push length)],
MIN_ROWS_PER_BIN=50 falls back to the identity operator for that bin.

Metric: `accuracy = 1 - rms(model)/rms(persistence)` on the SWEPT REGION,
via `fit_linear_foresight.swept_region_mask` + `.metrics` (the exact function
`Baselines/LinearForesight/fit_switched.py::evaluate` uses for its headline
number) -- named explicitly here since the task calls for stating exactly
which function was reused rather than inventing a new mask.
"""
from __future__ import annotations

import json
import sys
import time

import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")

from fit_linear_foresight import (  # noqa: E402
    actions_to_pixels, fit_operator, metrics, swept_region_mask,
)
from transforms.functional import (  # noqa: E402
    blend_push_prediction, from_push_frame, push_frame_validity_mask,
)
from descriptors import BOUNDS, GRID  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 32
N_BINS = 6
MIN_ROWS_PER_BIN = 50
LAMS = [0.1, 1.0, 10.0]
CACHE_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"

WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])

CELLS = {
    "visual":   dict(use_visual=True,  desc_dim=0),
    "hybrid14": dict(use_visual=True,  desc_dim=14),
    "hybrid94": dict(use_visual=True,  desc_dim=94),
    "desc14":   dict(use_visual=False, desc_dim=14),
    "desc94":   dict(use_visual=False, desc_dim=94),
}


def bin_index(lengths_m: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(lengths_m, bin_edges[1:-1].to(lengths_m.device))


def load_cache(split):
    d = torch.load(f"{CACHE_DIR}/{split}_cache.pt", map_location="cpu")
    N = d["canon0"].shape[0]
    d["canon0f"] = d["canon0"].reshape(N, -1)
    d["canon1f"] = d["canon1"].reshape(N, -1)
    s_px, e_px = actions_to_pixels(d["action"], WS_MIN, WS_MAX, (GRID, GRID))
    d["start_px"], d["end_px"] = s_px, e_px
    return d


def build_state(cache, cell, which):
    canon = cache[f"canon{which}f"]
    desc = cache[f"desc{which}"]
    dd = cell["desc_dim"]
    if cell["use_visual"] and dd > 0:
        return torch.cat([canon, desc[:, :dd]], dim=1)
    if cell["use_visual"]:
        return canon
    return desc[:, :dd]


def fit_one_bin(X0, X1, lam, D):
    if X0.shape[0] < MIN_ROWS_PER_BIN:
        return torch.eye(D)
    Y0, Y1 = X0.to(DEVICE).T, X1.to(DEVICE).T  # (D, n)
    A = fit_operator(Y0, Y1, ridge=lam, toward_identity=True)
    return A.cpu()


def fit_cell(cell, X0_tr, X1_tr, lengths_tr, bin_edges, lam):
    """Returns (operators list per bin, single global operator)."""
    D = X0_tr.shape[1]
    bins = bin_index(lengths_tr, bin_edges)
    ops = []
    for b in range(N_BINS):
        m = bins == b
        ops.append(fit_one_bin(X0_tr[m], X1_tr[m], lam, D))
    A_single = fit_one_bin(X0_tr, X1_tr, lam, D)
    return ops, A_single


def predict_image(cell, A_or_ops, switched, cache, bin_edges, batch=512):
    """Predict occ1 in world (64,64) frame for cell/operator choice.
    Descriptor-only cells cannot reconstruct an image at all -- returns occ0
    (exact persistence) by construction."""
    occ0 = cache["occ0"].float()
    if not cell["use_visual"]:
        return occ0.clone()

    canon0 = cache["canon0f"]
    desc0 = cache["desc0"]
    s_px, e_px = cache["start_px"], cache["end_px"]
    lengths = cache["length_m"]
    N = occ0.shape[0]
    H, W = 64, 64
    dd = cell["desc_dim"]
    out = torch.empty(N, H, W)
    bins = bin_index(lengths, bin_edges) if switched else None
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        c0 = canon0[sl].to(DEVICE)
        if dd > 0:
            x0 = torch.cat([c0, desc0[sl, :dd].to(DEVICE)], dim=1)
        else:
            x0 = c0
        s, e = s_px[sl].to(DEVICE), e_px[sl].to(DEVICE)
        o0 = occ0[sl].to(DEVICE)
        if switched:
            pred_c = torch.empty(x0.shape[0], RES * RES, device=DEVICE)
            b_sl = bins[sl]
            for b in range(N_BINS):
                m = b_sl == b
                if not bool(m.any()):
                    continue
                A = A_or_ops[b].to(DEVICE)
                pred_c[m] = (A @ x0[m].T).T[:, :RES * RES]
        else:
            A = A_or_ops.to(DEVICE)
            pred_c = (A @ x0.T).T[:, :RES * RES]
        canon1_hat = pred_c.reshape(-1, RES, RES)
        back = from_push_frame(canon1_hat, s, e, (H, W), 1.0)
        mask = push_frame_validity_mask(s, e, (H, W), (RES, RES), 1.0)
        pred = blend_push_prediction(back, o0, mask).clamp_(0.0, 1.0)
        out[sl] = pred.cpu()
    return out


def predict_desc(cell, A_or_ops, switched, cache, bin_edges, batch=2048):
    """Secondary metric: predicted descriptor vector at t+1 (raw, not image)."""
    dd = cell["desc_dim"]
    desc0 = cache["desc0"][:, :dd]
    lengths = cache["length_m"]
    N = desc0.shape[0]
    out = torch.empty(N, dd)
    bins = bin_index(lengths, bin_edges) if switched else None
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        x0 = desc0[sl].to(DEVICE)
        if cell["use_visual"]:
            x0 = torch.cat([cache["canon0f"][sl].to(DEVICE), x0], dim=1)
        if switched:
            pred = torch.empty(x0.shape[0], dd, device=DEVICE)
            b_sl = bins[sl]
            for b in range(N_BINS):
                m = b_sl == b
                if not bool(m.any()):
                    continue
                A = A_or_ops[b].to(DEVICE)
                full = (A @ x0[m].T).T
                pred[m] = full[:, -dd:] if cell["use_visual"] else full
        else:
            A = A_or_ops.to(DEVICE)
            full = (A @ x0.T).T
            pred = full[:, -dd:] if cell["use_visual"] else full
        out[sl] = pred.cpu()
    return out


def eval_accuracy(pred_img, cache, region):
    return metrics(pred_img, cache["occ1"].float(), cache["occ0"].float(), region=region)


def per_bin_accuracy(pred_img, cache, region, bin_edges):
    bins = bin_index(cache["length_m"], bin_edges)
    out = {}
    for b in range(N_BINS):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            out[b] = None
            continue
        out[b] = dict(n=n, **eval_accuracy(pred_img[m], {k: v[m] for k, v in cache.items()
                                                          if k in ("occ0", "occ1")}, region[m]))
    return out


def desc_accuracy(pred_desc, truth1, truth0):
    """Same accuracy FORMULA (1 - rms(err)/rms(persistence err)) applied
    directly to raw (non-z-scored) descriptor vectors -- NOT stage 1's exact
    z-scored/const-excluded pooled metric (`metric.py`), just the repo's
    standard accuracy formula reused as a cheap secondary number."""
    d = pred_desc - truth1
    p = truth0 - truth1
    rms_m = d.pow(2).mean(dim=1).sqrt().mean()
    rms_p = p.pow(2).mean(dim=1).sqrt().mean().clamp_min(1e-9)
    return float(1.0 - rms_m / rms_p)


def main():
    t0 = time.time()
    print("loading caches...")
    train = load_cache("train")
    test = load_cache("test")
    print(f"train N={train['canon0'].shape[0]}, test N={test['canon0'].shape[0]}, "
          f"{time.time()-t0:.1f}s")

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    print(f"push length range (train): min {float(train['length_m'].min())*1000:.2f} mm, "
          f"max {hi*1000:.2f} mm")

    plate_px = 0.04 / 0.128 * 64
    region_tr = swept_region_mask(train["start_px"], train["end_px"], (64, 64),
                                   0.5 * plate_px + 2.0, 0.5 * plate_px)
    region_te = swept_region_mask(test["start_px"], test["end_px"], (64, 64),
                                   0.5 * plate_px + 2.0, 0.5 * plate_px)

    results = {"bin_edges_mm": (bin_edges * 1000).tolist(), "cells": {}}

    # persistence (mandatory, cheap, no lam dependence)
    pers_tr = eval_accuracy(train["occ0"].float(), train, region_tr)
    pers_te = eval_accuracy(test["occ0"].float(), test, region_te)
    results["cells"]["persistence"] = {"train": pers_tr, "holdout": pers_te}
    print(f"persistence: train acc {pers_tr['accuracy']:.4f}  holdout acc {pers_te['accuracy']:.4f}")

    unswitched_cells = {"visual", "hybrid14", "hybrid94"}

    for cell_name, cell in CELLS.items():
        results["cells"][cell_name] = {}
        X0_tr = build_state(train, cell, "0")
        X1_tr = build_state(train, cell, "1")
        for lam in LAMS:
            t1 = time.time()
            ops, A_single = fit_cell(cell, X0_tr, X1_tr, train["length_m"], bin_edges, lam)
            entry = {}
            for split_name, cache, region in [("train", train, region_tr),
                                               ("holdout", test, region_te)]:
                pred_sw = predict_image(cell, ops, True, cache, bin_edges)
                acc_sw = eval_accuracy(pred_sw, cache, region)
                pb_sw = per_bin_accuracy(pred_sw, cache, region, bin_edges)
                entry[split_name] = {"switched": acc_sw, "per_bin_switched": pb_sw}

                if cell_name in unswitched_cells:
                    pred_gl = predict_image(cell, A_single, False, cache, bin_edges)
                    acc_gl = eval_accuracy(pred_gl, cache, region)
                    pb_gl = per_bin_accuracy(pred_gl, cache, region, bin_edges)
                    entry[split_name]["global"] = acc_gl
                    entry[split_name]["per_bin_global"] = pb_gl

                # secondary descriptor-space metric
                if cell["desc_dim"] > 0:
                    pred_d_sw = predict_desc(cell, ops, True, cache, bin_edges)
                    truth1 = cache["desc1"][:, :cell["desc_dim"]]
                    truth0 = cache["desc0"][:, :cell["desc_dim"]]
                    entry[split_name]["desc_accuracy_switched"] = desc_accuracy(
                        pred_d_sw, truth1, truth0)
            results["cells"][cell_name][str(lam)] = entry
            print(f"  [{cell_name} lam={lam}] train acc(sw)={entry['train']['switched']['accuracy']:.4f} "
                  f"holdout acc(sw)={entry['holdout']['switched']['accuracy']:.4f}  "
                  f"({time.time()-t1:.1f}s)")

    out_path = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/results_stage2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
