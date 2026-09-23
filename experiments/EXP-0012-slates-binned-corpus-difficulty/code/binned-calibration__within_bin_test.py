"""experiments/temp/binned-calibration/within_bin_test.py

Follow-up to calibrate.py: tests the REAL version of the "cross-bin
calibration" concern flagged in RESULTS.md. The originally-stated version
(MODEL-0002's per-bin operator assigns a bin per POOL rather than per
candidate) is wrong -- predict_switched (Baselines/LinearForesight/model.py
and desc-mlp/eval_control.py) both bucketize length_m per-row, so each
candidate already gets its own bin's operator.

The real version: because the 6 bin-operators are fit INDEPENDENTLY, their
prediction biases are independent of each other. Ranking candidates whose
predictions come from DIFFERENT operators (as slates_binned pools do -- 5
length bins mixed within every slate's 1000-candidate pool) compares
differently-calibrated estimates in a way that never arises on
slates_multistep (single push length per corpus -> one operator per pool).

Test: restrict each slate's pool to candidates in a SINGLE bin (do this per
well-populated bin, separately), rescore switched-linear slateN there, and
compare to the full (across-bin) pool's slateN already measured in
calibrate.py's results_calibration.json (K=32/K=128 fixed refs, corner
goal, lyapunov value fn -- same setup, for apples-to-apples).

If within-bin capture is clearly better than across-bin: cross-bin
calibration is a real contributor, and the "difficulty-only" verdict in
RESULTS.md needs softening (and the bin explanation, corrected to be
per-candidate cross-bin comparison rather than per-pool, should be kept).
If not: the difficulty explanation stands and the bin wording should be
struck outright.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/desc-mlp")

from control_utility_test import lyapunov, lyapunov_weights  # noqa: E402
from Baselines.common.goals import slate_n_capture  # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus, UNDERFLOW_BIN  # noqa: E402
from eval_control import (  # noqa: E402
    compute_descriptors, predict_switched, load_switched, com_world_pixel, bilinear_sample,
)
from Baselines.LinearForesight.model import bin_index as bin_index_switch  # noqa: E402

DEVICE = "cpu"
K_FIXED = 32
N_RESAMPLE = 150


def corner_goal(H, W):
    dw = lyapunov_weights((H, W), "corner", DEVICE)
    return dw


def score_pool_kfixed(v_true, v_pred, slate_ids, higher, k, n_resample=N_RESAMPLE, seed=0, min_pool=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        n = rows.numel()
        if n < (min_pool or k):
            continue
        vt, vp = v_true[rows], v_pred[rows]
        per = []
        kk = min(k, n)
        for _ in range(n_resample):
            idx = torch.randperm(n, generator=g)[:kk]
            c = slate_n_capture(vp[idx], vt[idx], higher)
            if c == c:
                per.append(c)
        if per:
            out.append(float(np.mean(per)))
    return out


def score_pool_exact(v_true, v_pred, slate_ids, higher):
    caps = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        c = slate_n_capture(v_pred[rows], v_true[rows], higher)
        if c == c:
            caps.append(c)
    return caps


def sem(x):
    x = np.asarray(x)
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    print("[1/2] loading MODEL-0002 switched-linear operator + slates_binned...")
    sw_ops, sw_edges, sw_slices = load_switched()
    print("  bin_edges (mm):", (sw_edges * 1000).tolist())

    corpus = BinnedSlateCorpus.load("Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm")
    rows = corpus.step(0)
    states = rows.states[:, :, :3].float()
    states_ = rows.states_[:, :, :3].float()
    p_start = rows.p_starts[:, :2].float()
    p_stop = rows.p_stops[:, :2].float()
    slate_idx = rows.slate_idx.long()
    # NOTE: corpus's own `bin_realized` uses BinnedSlateCorpus's collection-time
    # 5-bin scheme over 20-70mm -- NOT the same bin edges MODEL-0002's operator
    # switches on (6 bins, 0-80mm, from EXP-0003/fit_switched.py). The question
    # here is "does restricting to candidates that get the SAME OPERATOR help",
    # so we must group by MODEL-0002's own bin assignment (bin_index_switch on
    # length_m), not the corpus's collection-time bin_realized field.
    bin_realized = None  # computed below from length_m + sw_edges instead

    occ0, occ1, phi0, length_m, s_px, e_px = compute_descriptors(states, states_, p_start, p_stop)
    H, W = occ0.shape[-2:]
    dw = corner_goal(H, W)
    v_true = lyapunov(occ1, dw)
    v_pre = lyapunov(occ0, dw)

    phi_sw = predict_switched(sw_ops, sw_edges, phi0, length_m)
    com = phi_sw[:, 3:5]
    wr_sw, wc_sw = com_world_pixel(com[:, 0], com[:, 1], s_px, e_px, H, W)
    v_sw = bilinear_sample(dw.cpu(), wr_sw, wc_sw)

    bin_realized = bin_index_switch(length_m, sw_edges)  # MODEL-0002's OWN operator-bin assignment
    print("MODEL-0002 operator-bin value counts (per-candidate):",
          {int(b): int((bin_realized == b).sum()) for b in bin_realized.unique().tolist()})

    # --- across-bin (full pool, mixed bins) reference, matching calibrate.py exactly ---
    across_32 = score_pool_kfixed(v_true, v_sw, slate_idx, False, K_FIXED)
    across_128 = score_pool_kfixed(v_true, v_sw, slate_idx, False, 128)
    print(f"[full pool / across-bin] K=32: {np.mean(across_32):.4f} (sem {sem(across_32):.4f}, n={len(across_32)})")
    print(f"[full pool / across-bin] K=128: {np.mean(across_128):.4f} (sem {sem(across_128):.4f}, n={len(across_128)})")

    # --- within-bin: restrict to one bin at a time, per slate ---
    print("[2/2] within-bin (single-operator pools)...")
    per_bin = {}
    all_within_32 = []
    all_within_exact = []
    for b in sorted(bin_realized.unique().tolist()):
        sel = (bin_realized == b)
        n_total = int(sel.sum())
        if n_total == 0:
            continue
        v_true_b = v_true[sel]
        v_sw_b = v_sw[sel]
        slate_idx_b = slate_idx[sel]
        # per-slate pool size within this bin
        pool_sizes = [int((slate_idx_b == s).sum()) for s in slate_idx_b.unique().tolist()]
        min_pool = min(pool_sizes) if pool_sizes else 0
        exact = score_pool_exact(v_true_b, v_sw_b, slate_idx_b, False)
        k32 = score_pool_kfixed(v_true_b, v_sw_b, slate_idx_b, False, min(K_FIXED, min_pool) if min_pool else K_FIXED,
                                 min_pool=1)
        per_bin[b] = {
            "n_rows": n_total, "n_slates_with_data": len(pool_sizes),
            "min_pool_per_slate": min_pool, "max_pool_per_slate": max(pool_sizes) if pool_sizes else 0,
            "mean_pool_per_slate": float(np.mean(pool_sizes)) if pool_sizes else 0,
            "slateN_exact_mean": float(np.mean(exact)) if exact else float("nan"),
            "slateN_exact_sem": sem(exact), "n_exact": len(exact),
            "slateN_k_capped_mean": float(np.mean(k32)) if k32 else float("nan"),
            "slateN_k_capped_sem": sem(k32), "n_k_capped": len(k32),
        }
        all_within_exact += exact
        if min_pool >= K_FIXED:
            all_within_32 += k32
        print(f"  bin {b}: n_rows={n_total}, slates={len(pool_sizes)}, "
              f"mean_pool/slate={np.mean(pool_sizes):.1f}, "
              f"slateN(exact)={per_bin[b]['slateN_exact_mean']:.4f} (sem {per_bin[b]['slateN_exact_sem']:.4f})")

    pooled_within_exact = {"mean": float(np.mean(all_within_exact)), "sem": sem(all_within_exact), "n": len(all_within_exact)}
    print(f"\nPOOLED within-bin (exact K=N per bin, pooled over bins/slates): "
          f"{pooled_within_exact['mean']:.4f} (sem {pooled_within_exact['sem']:.4f}, n={pooled_within_exact['n']})")
    print(f"POOLED across-bin (full 1000-candidate pool) K=32 fixed ref: "
          f"{np.mean(across_32):.4f} (sem {sem(across_32):.4f}, n={len(across_32)})")

    results = {
        "bin_edges_mm": (sw_edges * 1000).tolist(),
        "across_bin_full_pool": {
            "K32": {"mean": float(np.mean(across_32)), "sem": sem(across_32), "n": len(across_32)},
            "K128": {"mean": float(np.mean(across_128)), "sem": sem(across_128), "n": len(across_128)},
        },
        "within_bin": per_bin,
        "within_bin_pooled_exact": pooled_within_exact,
    }
    with open("/home/alon/Code/pile_manipulation/experiments/temp/binned-calibration/results_within_bin.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote results_within_bin.json")


if __name__ == "__main__":
    main()
