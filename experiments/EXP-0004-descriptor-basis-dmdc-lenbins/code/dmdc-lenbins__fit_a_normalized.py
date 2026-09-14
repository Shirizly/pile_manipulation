"""Re-score variant A under the fixed z-scored / const-dropped metric (metric.py).

Same ridge fits as fit.py (same bins, same lams) -- only the ACCURACY METRIC
changes, so this is directly comparable to variant B's numbers computed the
same way.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from dmdc_baseline import fit_per_action_operators, apply_operators, descriptor_slices
from metric import fit_zscore, descriptor_accuracy

CACHE = "experiments/temp/dmdc-lenbins/cache"
N_BINS = 6
MIN_ROWS_PER_BIN = 50
N_FOURIER = 8
LAMS = [1e-4, 1e-3, 1e-2]


def load(split):
    d = np.load(f"{CACHE}/desc_{split}.npz", allow_pickle=True)
    return {
        "phi_t": torch.from_numpy(d["phi_t"]),
        "phi_t1": torch.from_numpy(d["phi_t1"]),
        "length_m": torch.from_numpy(d["length_m"]),
        "spawn_mode": torch.from_numpy(d["spawn_mode"]),
        "file_id": torch.from_numpy(d["file_id"]),
    }


def bin_index(lengths_m, bin_edges):
    return torch.bucketize(lengths_m, bin_edges[1:-1])


def main():
    train = load("train")
    test = load("test")
    slices = descriptor_slices(N_FOURIER)
    D = slices["_total"].stop
    mu, sigma = fit_zscore(train["phi_t1"])

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    bins_train = bin_index(train["length_m"], bin_edges)
    bins_test = bin_index(test["length_m"], bin_edges)

    results = {"bin_edges_mm": (bin_edges * 1000).tolist(), "n_bins": N_BINS,
               "min_rows_per_bin": MIN_ROWS_PER_BIN, "metric": "zscored_no_const", "lam": {}}

    for lam in LAMS:
        A_sw, counts = fit_per_action_operators(train["phi_t"], train["phi_t1"], bins_train,
                                                  N_BINS, lam=lam)
        for b in range(N_BINS):
            if int(counts[b]) < MIN_ROWS_PER_BIN:
                A_sw[b] = torch.eye(D)
        bins_all0 = torch.zeros_like(bins_train)
        A_gl, _ = fit_per_action_operators(train["phi_t"], train["phi_t1"], bins_all0, 1, lam=lam)

        pred_sw = apply_operators(A_sw, test["phi_t"], bins_test)
        pred_gl = apply_operators(A_gl, test["phi_t"], torch.zeros_like(bins_test))
        persist = test["phi_t"]
        truth = test["phi_t1"]

        r_sw = descriptor_accuracy(pred_sw, truth, persist, mu, sigma, slices)
        r_gl = descriptor_accuracy(pred_gl, truth, persist, mu, sigma, slices)

        per_bin = {}
        for b in range(N_BINS):
            m = bins_test == b
            nb = int(m.sum())
            if nb == 0:
                per_bin[b] = {"n_test": 0}
                continue
            a_sw = descriptor_accuracy(pred_sw[m], truth[m], persist[m], mu, sigma, slices)["overall"]
            a_gl = descriptor_accuracy(pred_gl[m], truth[m], persist[m], mu, sigma, slices)["overall"]
            per_bin[b] = {
                "n_train": int(counts[b]), "n_test": nb,
                "descriptor_accuracy_switched": a_sw,
                "descriptor_accuracy_global": a_gl,
                "lo_mm": float(bin_edges[b]) * 1000, "hi_mm": float(bin_edges[b + 1]) * 1000,
            }

        print(f"lam={lam}: overall switched={r_sw['overall']:.4f} global={r_gl['overall']:.4f}")
        print("  blocks (switched):", {k: round(v, 4) for k, v in r_sw["blocks"].items()})

        results["lam"][str(lam)] = {
            "overall_descriptor_accuracy_switched": r_sw["overall"],
            "overall_descriptor_accuracy_global": r_gl["overall"],
            "block_descriptor_accuracy_switched": r_sw["blocks"],
            "block_descriptor_accuracy_global": r_gl["blocks"],
            "per_bin": per_bin,
            "train_bin_counts": counts.tolist(),
        }

    outpath = "experiments/temp/dmdc-lenbins/results_variantA_normalized.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
