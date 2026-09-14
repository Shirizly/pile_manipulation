"""Common-target ablation: hold the TARGET fixed (D-local-minus-DFT's 14-dim
vector at t+1, exactly as descriptors_d.slices_d_no_dft lays it out) and vary
only the INPUT feature set. This is the ablation stage 1's pooled-headline
comparison (A vs B vs D-all-local vs D-no-dft vs control) never actually ran:
that comparison pooled `1 - rms(model)/rms(persistence)` over EACH variant's
OWN dimension set, so cells with fewer (better-predicted) target dims looked
better partly because the denominator changed, not because the features did.

Fixed target for every cell: descD cache's phiD_t1[:, :14] (global_mass,
global_length, mass, com(2), moments2(3), mass_ahead, mass_behind, band_mass(4)
-- descriptors_d.slices_d_no_dft's layout). global_length is degenerate
(identical at t/t+1 by construction, a property of the action not the state)
and excluded from the pooled ratio exactly as fit_d.py excludes it.

Persistence baseline is ALWAYS target_t (the same fixed 14-dim vector at time
t), regardless of which input feature set is being fit -- this is what makes
the six cells comparable: same numerator target, same denominator baseline,
only the regressor's INPUT columns change.

Cells (input feature set -> fixed 14-dim target):
  1. stock_global_A   : variant A's 87 stock global descriptors (phi_t, desc cache)
  2. pushframe_B       : variant B's 11 push-frame-only dims (phiB_t)
  3. D_nodft_self      : the target's own 14 dims at time t (phiD_t[:, :14])
  4. D_all_local       : D-all-local's 94 dims (phiD_t, incl. DFT block)
  5. A_plus_B          : A's 87 + B's 11 concatenated (98)
  6. persistence       : mandatory do-nothing control, pred = target_t exactly
                         (not fit -- this is 0.0 by construction of the metric,
                         reported for completeness / sanity, not a real cell)

Ridge fit is a rectangular per-bin ridge (D_in -> D_out=14), the same normal
equations as dmdc_baseline.fit_per_action_operators generalised off the square
case (that function assumes D_in==D_out so it can't be reused directly here).
Same 6 length bins (EXP-0003 scheme), same 80/20 file split seed 0 (baked into
the caches already), same ridge lam sweep, same switched vs single-global-
operator fits, train AND holdout reported -- everything else held identical to
variant D's own fit_d.py.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from metric import fit_zscore, descriptor_accuracy
from descriptors_d import slices_d_no_dft

CACHE = "experiments/temp/dmdc-lenbins/cache"
N_BINS = 6
MIN_ROWS_PER_BIN = 50
LAMS = [1e-4, 1e-3, 1e-2]
EXCLUDE = {"const", "global_length"}


def load(split):
    d_a = np.load(f"{CACHE}/desc_{split}.npz", allow_pickle=True)
    d_b = np.load(f"{CACHE}/descB_{split}.npz", allow_pickle=True)
    d_d = np.load(f"{CACHE}/descD_{split}.npz", allow_pickle=True)
    phi_a = torch.from_numpy(d_a["phi_t"].astype(np.float32))
    phi_b_t = torch.from_numpy(d_b["phiB_t"].astype(np.float32))
    phi_d_t = torch.from_numpy(d_d["phiD_t"].astype(np.float32))
    phi_d_t1 = torch.from_numpy(d_d["phiD_t1"].astype(np.float32))
    length_m = torch.from_numpy(d_a["length_m"].astype(np.float32))
    return dict(phi_a=phi_a, phi_b_t=phi_b_t, phi_d_t=phi_d_t, phi_d_t1=phi_d_t1,
                length_m=length_m)


def bin_index(lengths_m, bin_edges):
    return torch.bucketize(lengths_m, bin_edges[1:-1])


def fit_rect_per_bin(X_in: torch.Tensor, Y_out: torch.Tensor, bins: torch.Tensor,
                      n_bins: int, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular per-bin ridge: A_b [Dout,Din] minimising
    ||A_b X - Y||^2 + lam||A_b||^2, solved as A = Y X^T (X X^T + lam I)^-1.
    Empty bins fall back to a zero operator (there is no sensible "identity"
    when Din != Dout); MIN_ROWS_PER_BIN-thin bins are flagged by caller same
    as fit_d.py's convention if desired -- kept simple here since no bin was
    ever under 50 rows in this dataset (see fit_d.py's own count table)."""
    N, Din = X_in.shape
    Dout = Y_out.shape[1]
    A = torch.zeros(n_bins, Dout, Din)
    counts = torch.zeros(n_bins, dtype=torch.long)
    I = torch.eye(Din, dtype=torch.float64)
    for b in range(n_bins):
        mask = bins == b
        nb = int(mask.sum())
        counts[b] = nb
        if nb == 0:
            continue
        X = X_in[mask].T.double()   # [Din, nb]
        Y = Y_out[mask].T.double()  # [Dout, nb]
        At = torch.linalg.solve(X @ X.T + lam * I, X @ Y.T)  # [Din, Dout]
        A[b] = At.T.float()
    return A, counts


def apply_rect(A: torch.Tensor, X_in: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return torch.bmm(A[bins], X_in.unsqueeze(-1)).squeeze(-1)


def run_cell(name, X_tr, X_te, target_t_tr, target_t1_tr, target_t_te, target_t1_te,
             len_tr, len_te, slices, persistence_only=False):
    mu, sigma = fit_zscore(target_t1_tr)
    hi = float(len_tr.max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    bins_tr = bin_index(len_tr, bin_edges)
    bins_te = bin_index(len_te, bin_edges)

    cell = {"name": name, "D_in": (0 if persistence_only else X_tr.shape[1]),
            "bin_edges_mm": (bin_edges * 1000).tolist(), "n_bins": N_BINS, "lam": {}}

    for lam in LAMS:
        if persistence_only:
            def score(target_t, target_t1, bins):
                pred = target_t  # do-nothing
                r = descriptor_accuracy(pred, target_t1, target_t, mu, sigma, slices, exclude_blocks=EXCLUDE)
                per_bin = {}
                for b in range(N_BINS):
                    m = bins == b
                    nb = int(m.sum())
                    if nb == 0:
                        per_bin[b] = {"n": 0}
                        continue
                    a = descriptor_accuracy(pred[m], target_t1[m], target_t[m], mu, sigma, slices, exclude_blocks=EXCLUDE)["overall"]
                    per_bin[b] = {"n": nb, "value": a}
                return r, r, per_bin

            r_sw_tr, r_gl_tr, per_bin_tr = score(target_t_tr, target_t1_tr, bins_tr)
            r_sw_te, r_gl_te, per_bin_te = score(target_t_te, target_t1_te, bins_te)
            counts = torch.zeros(N_BINS, dtype=torch.long)
        else:
            A_sw, counts = fit_rect_per_bin(X_tr, target_t1_tr, bins_tr, N_BINS, lam)
            for b in range(N_BINS):
                if int(counts[b]) < MIN_ROWS_PER_BIN:
                    print(f"[{name}] lam={lam} bin {b}: only {int(counts[b])} rows (<{MIN_ROWS_PER_BIN})")
            zeros_tr = torch.zeros_like(bins_tr)
            A_gl, _ = fit_rect_per_bin(X_tr, target_t1_tr, zeros_tr, 1, lam)

            def score(X, target_t, target_t1, bins, zeros):
                pred_sw = apply_rect(A_sw, X, bins)
                pred_gl = apply_rect(A_gl, X, zeros)
                r_sw = descriptor_accuracy(pred_sw, target_t1, target_t, mu, sigma, slices, exclude_blocks=EXCLUDE)
                r_gl = descriptor_accuracy(pred_gl, target_t1, target_t, mu, sigma, slices, exclude_blocks=EXCLUDE)
                per_bin = {}
                for b in range(N_BINS):
                    m = bins == b
                    nb = int(m.sum())
                    if nb == 0:
                        per_bin[b] = {"n": 0}
                        continue
                    a_sw = descriptor_accuracy(pred_sw[m], target_t1[m], target_t[m], mu, sigma, slices, exclude_blocks=EXCLUDE)["overall"]
                    a_gl = descriptor_accuracy(pred_gl[m], target_t1[m], target_t[m], mu, sigma, slices, exclude_blocks=EXCLUDE)["overall"]
                    per_bin[b] = {"n": nb, "switched": a_sw, "global": a_gl}
                return r_sw, r_gl, per_bin

            r_sw_tr, r_gl_tr, per_bin_tr = score(X_tr, target_t_tr, target_t1_tr, bins_tr, zeros_tr)
            r_sw_te, r_gl_te, per_bin_te = score(X_te, target_t_te, target_t1_te, bins_te, torch.zeros_like(bins_te))

        print(f"[{name}] lam={lam}: train switched={r_sw_tr['overall']:.4f} global={r_gl_tr['overall']:.4f} | "
              f"holdout switched={r_sw_te['overall']:.4f} global={r_gl_te['overall']:.4f}")

        cell["lam"][str(lam)] = {
            "train_bin_counts": counts.tolist(),
            "train": {"overall_switched": r_sw_tr["overall"], "overall_global": r_gl_tr["overall"],
                       "blocks_switched": r_sw_tr["blocks"], "per_bin": per_bin_tr},
            "holdout": {"overall_switched": r_sw_te["overall"], "overall_global": r_gl_te["overall"],
                        "blocks_switched": r_sw_te["blocks"], "per_bin": per_bin_te},
        }
    return cell


def main():
    tr = load("train")
    te = load("test")
    slices = slices_d_no_dft()
    cut = slices["_total"].stop
    assert cut == 14, cut

    target_t_tr, target_t1_tr = tr["phi_d_t"][:, :cut], tr["phi_d_t1"][:, :cut]
    target_t_te, target_t1_te = te["phi_d_t"][:, :cut], te["phi_d_t1"][:, :cut]
    len_tr, len_te = tr["length_m"], te["length_m"]

    results = {}

    results["stock_global_A"] = run_cell(
        "stock_global_A", tr["phi_a"], te["phi_a"],
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices)

    results["pushframe_B"] = run_cell(
        "pushframe_B", tr["phi_b_t"], te["phi_b_t"],
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices)

    results["D_nodft_self"] = run_cell(
        "D_nodft_self", tr["phi_d_t"][:, :cut], te["phi_d_t"][:, :cut],
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices)

    results["D_all_local"] = run_cell(
        "D_all_local", tr["phi_d_t"], te["phi_d_t"],
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices)

    X_ab_tr = torch.cat([tr["phi_a"], tr["phi_b_t"]], dim=1)
    X_ab_te = torch.cat([te["phi_a"], te["phi_b_t"]], dim=1)
    results["A_plus_B"] = run_cell(
        "A_plus_B", X_ab_tr, X_ab_te,
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices)

    results["persistence"] = run_cell(
        "persistence", None, None,
        target_t_tr, target_t1_tr, target_t_te, target_t1_te, len_tr, len_te, slices,
        persistence_only=True)

    outpath = "experiments/temp/dmdc-lenbins/results_arbitration.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
