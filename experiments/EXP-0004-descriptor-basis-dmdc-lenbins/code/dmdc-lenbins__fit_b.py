"""Variant B: variant-A descriptor vector (D=87) concatenated with push-frame
local descriptors (D=11, descriptors_b.py) -> D=98. Same switched-linear DMDc
fit (same 6 length bins, same ridge sweep) as variant A, scored with the
SAME fixed metric (metric.py: z-score by train phi_t1, const dropped).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from dmdc_baseline import fit_per_action_operators, apply_operators, descriptor_slices
from metric import fit_zscore, descriptor_accuracy
from descriptors_b import slices_b

CACHE = "experiments/temp/dmdc-lenbins/cache"
N_BINS = 6
MIN_ROWS_PER_BIN = 50
N_FOURIER = 8
LAMS = [1e-4, 1e-3, 1e-2]


def merged_slices():
    """Variant-A block layout followed by variant-B blocks, offset by D_A."""
    sl_a = descriptor_slices(N_FOURIER)
    D_a = sl_a["_total"].stop
    sl_b = slices_b()
    D_b = sl_b["_total"].stop
    merged = {}
    for name, sl in sl_a.items():
        if name == "_total":
            continue
        merged[name] = sl
    for name, sl in sl_b.items():
        if name == "_total":
            continue
        merged[f"push_{name}"] = slice(sl.start + D_a, sl.stop + D_a)
    merged["_total"] = slice(0, D_a + D_b)
    return merged, D_a, D_b


def load(split):
    d = np.load(f"{CACHE}/desc_{split}.npz", allow_pickle=True)
    b = np.load(f"{CACHE}/descB_{split}.npz", allow_pickle=True)
    phi_t = np.concatenate([d["phi_t"], b["phiB_t"]], axis=1)
    phi_t1 = np.concatenate([d["phi_t1"], b["phiB_t1"]], axis=1)
    return {
        "phi_t": torch.from_numpy(phi_t.astype(np.float32)),
        "phi_t1": torch.from_numpy(phi_t1.astype(np.float32)),
        "length_m": torch.from_numpy(d["length_m"]),
        "spawn_mode": torch.from_numpy(d["spawn_mode"]),
        "file_id": torch.from_numpy(d["file_id"]),
    }


def bin_index(lengths_m, bin_edges):
    return torch.bucketize(lengths_m, bin_edges[1:-1])


def main():
    train = load("train")
    test = load("test")
    slices, D_a, D_b = merged_slices()
    D = slices["_total"].stop
    assert D == train["phi_t"].shape[1] == D_a + D_b
    mu, sigma = fit_zscore(train["phi_t1"])

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    bins_train = bin_index(train["length_m"], bin_edges)
    bins_test = bin_index(test["length_m"], bin_edges)

    results = {"bin_edges_mm": (bin_edges * 1000).tolist(), "n_bins": N_BINS,
               "min_rows_per_bin": MIN_ROWS_PER_BIN, "metric": "zscored_no_const",
               "D_total": D, "D_variantA": D_a, "D_pushframe": D_b, "lam": {}}

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

    outpath = "experiments/temp/dmdc-lenbins/results_variantB.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
