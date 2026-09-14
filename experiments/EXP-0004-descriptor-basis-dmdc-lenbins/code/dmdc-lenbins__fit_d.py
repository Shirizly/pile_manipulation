"""Variant D fit: three cells, all switched-linear DMDc (6 length bins, EXP-0003
scheme; same 80/20 file split seed 0; ridge lam in {1e-4,1e-3,1e-2}; metric.py
z-scored-by-train-phi_t1, const-free) so all three are directly comparable to
variant A/B.

  Cell 1 "D_all_local"        : descriptors_d.py full vector, D=94.
  Cell 2 "D_no_dft"           : same cache, DFT block dropped (a column
                                 prefix of cell 1's vector -- see
                                 descriptors_d.slices_d_no_dft), D=14.
  Cell 3 "control_pushonly11" : variant B's push-frame block ONLY (no
                                 variant-A global block at all), D=11 -- the
                                 honest dim-matched control: does D's richer
                                 local basis beat simply deleting the 87
                                 global dims from B?

Reports BOTH train and holdout descriptor_accuracy for every lam/bin, so
overfitting is visible (a large train>holdout gap would flag it).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from dmdc_baseline import fit_per_action_operators, apply_operators
from metric import fit_zscore, descriptor_accuracy
from descriptors_d import slices_d, slices_d_no_dft
from descriptors_b import slices_b

CACHE = "experiments/temp/dmdc-lenbins/cache"
N_BINS = 6
MIN_ROWS_PER_BIN = 50
LAMS = [1e-4, 1e-3, 1e-2]


def load_lengths(split):
    d = np.load(f"{CACHE}/desc_{split}.npz", allow_pickle=True)
    return torch.from_numpy(d["length_m"])


def load_D(split):
    d = np.load(f"{CACHE}/descD_{split}.npz", allow_pickle=True)
    return (torch.from_numpy(d["phiD_t"].astype(np.float32)),
            torch.from_numpy(d["phiD_t1"].astype(np.float32)))


def load_B_pushonly(split):
    d = np.load(f"{CACHE}/descB_{split}.npz", allow_pickle=True)
    return (torch.from_numpy(d["phiB_t"].astype(np.float32)),
            torch.from_numpy(d["phiB_t1"].astype(np.float32)))


def bin_index(lengths_m, bin_edges):
    return torch.bucketize(lengths_m, bin_edges[1:-1])


def run_cell(name, phi_t_tr, phi_t1_tr, phi_t_te, phi_t1_te, len_tr, len_te, slices):
    D = phi_t_tr.shape[1]
    assert phi_t1_tr.shape[1] == D and phi_t_te.shape[1] == D and phi_t1_te.shape[1] == D
    mu, sigma = fit_zscore(phi_t1_tr)
    # global_length is duplicated identically at t and t+1 by construction
    # (it's a property of the ACTION, not the state) -- persistence error on
    # it is exactly 0, so it is degenerate under the ratio metric exactly
    # like variant A's `const` block. Exclude both from the pooled/overall
    # number and treat its per-block entry as garbage, same convention as
    # metric.py's own const handling.
    exclude = {"const", "global_length"}

    hi = float(len_tr.max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    bins_tr = bin_index(len_tr, bin_edges)
    bins_te = bin_index(len_te, bin_edges)

    cell = {"name": name, "D": D, "bin_edges_mm": (bin_edges * 1000).tolist(),
            "n_bins": N_BINS, "lam": {}}

    for lam in LAMS:
        A_sw, counts = fit_per_action_operators(phi_t_tr, phi_t1_tr, bins_tr, N_BINS, lam=lam)
        for b in range(N_BINS):
            if int(counts[b]) < MIN_ROWS_PER_BIN:
                A_sw[b] = torch.eye(D)
        zeros_tr = torch.zeros_like(bins_tr)
        A_gl, _ = fit_per_action_operators(phi_t_tr, phi_t1_tr, zeros_tr, 1, lam=lam)

        def score(phi_t, phi_t1, bins, zeros):
            pred_sw = apply_operators(A_sw, phi_t, bins)
            pred_gl = apply_operators(A_gl, phi_t, zeros)
            persist = phi_t
            truth = phi_t1
            r_sw = descriptor_accuracy(pred_sw, truth, persist, mu, sigma, slices, exclude_blocks=exclude)
            r_gl = descriptor_accuracy(pred_gl, truth, persist, mu, sigma, slices, exclude_blocks=exclude)
            per_bin = {}
            for b in range(N_BINS):
                m = bins == b
                nb = int(m.sum())
                if nb == 0:
                    per_bin[b] = {"n": 0}
                    continue
                a_sw = descriptor_accuracy(pred_sw[m], truth[m], persist[m], mu, sigma, slices, exclude_blocks=exclude)["overall"]
                a_gl = descriptor_accuracy(pred_gl[m], truth[m], persist[m], mu, sigma, slices, exclude_blocks=exclude)["overall"]
                per_bin[b] = {"n": nb, "switched": a_sw, "global": a_gl}
            return r_sw, r_gl, per_bin

        r_sw_tr, r_gl_tr, per_bin_tr = score(phi_t_tr, phi_t1_tr, bins_tr, zeros_tr)
        r_sw_te, r_gl_te, per_bin_te = score(phi_t_te, phi_t1_te, bins_te, torch.zeros_like(bins_te))

        print(f"[{name}] lam={lam}: "
              f"train switched={r_sw_tr['overall']:.4f} global={r_gl_tr['overall']:.4f} | "
              f"holdout switched={r_sw_te['overall']:.4f} global={r_gl_te['overall']:.4f}")

        cell["lam"][str(lam)] = {
            "train_bin_counts": counts.tolist(),
            "train": {
                "overall_switched": r_sw_tr["overall"], "overall_global": r_gl_tr["overall"],
                "blocks_switched": r_sw_tr["blocks"], "per_bin": per_bin_tr,
            },
            "holdout": {
                "overall_switched": r_sw_te["overall"], "overall_global": r_gl_te["overall"],
                "blocks_switched": r_sw_te["blocks"], "per_bin": per_bin_te,
            },
        }
    return cell


def main():
    len_tr, len_te = load_lengths("train"), load_lengths("test")

    phiD_t_tr, phiD_t1_tr = load_D("train")
    phiD_t_te, phiD_t1_te = load_D("test")

    slices_full = slices_d()
    D_full = slices_full["_total"].stop
    cut = slices_d_no_dft()["_total"].stop

    results = {}

    results["D_all_local"] = run_cell(
        "D_all_local", phiD_t_tr, phiD_t1_tr, phiD_t_te, phiD_t1_te, len_tr, len_te, slices_full)

    slices_nodft = slices_d_no_dft()
    results["D_no_dft"] = run_cell(
        "D_no_dft", phiD_t_tr[:, :cut], phiD_t1_tr[:, :cut],
        phiD_t_te[:, :cut], phiD_t1_te[:, :cut], len_tr, len_te, slices_nodft)

    phiB_t_tr, phiB_t1_tr = load_B_pushonly("train")
    phiB_t_te, phiB_t1_te = load_B_pushonly("test")
    results["control_pushonly11"] = run_cell(
        "control_pushonly11", phiB_t_tr, phiB_t1_tr, phiB_t_te, phiB_t1_te,
        len_tr, len_te, slices_b())

    outpath = "experiments/temp/dmdc-lenbins/results_variantD.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
