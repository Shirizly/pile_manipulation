"""Fit switched-linear (by push-length bin) DMDc operators on cached descriptors
and evaluate descriptor_accuracy vs persistence, on held-out FILES.

Bin scheme: EXP-0003's (Baselines/LinearForesight/fit_switched.py ~L199):
    bin_edges = linspace(0, lengths.max(), n_bins+1), n_bins=6
    MIN_ROWS_PER_BIN = 50 -> below this, bin falls back to identity (persistence)
lengths.max() is taken from the TRAIN split only (edges must not see test data).

descriptor_accuracy = 1 - rms(model_error) / rms(persistence_error), in the
SAME normalised descriptor space top and bottom (raw dmdc_baseline.phi units;
no additional rescaling -- the DFT block is already O(mean occupancy) by
construction, see occupancy_descriptors docstring).

This is a PROXY for the repo's image-space `accuracy` metric (rms pixel
error in the swept region) -- it operates on the 87-d descriptor vector, not
reconstructed occupancy, so it can look better than image-space accuracy
would (moments can match while pixel-level shape does not) or worse (it has
no swept-region masking). Flagged in RESULTS.md too.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from dmdc_baseline import fit_per_action_operators, apply_operators, descriptor_slices

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


def rms(x):
    return float(torch.sqrt((x ** 2).mean()))


def block_rms(err, slices):
    out = {}
    for name, sl in slices.items():
        if name == "_total":
            continue
        out[name] = rms(err[:, sl])
    return out


def descriptor_accuracy(pred, truth, persist):
    model_err = pred - truth
    persist_err = persist - truth
    return 1.0 - rms(model_err) / max(rms(persist_err), 1e-12), model_err, persist_err


def main():
    train = load("train")
    test = load("test")
    slices = descriptor_slices(N_FOURIER)
    D = slices["_total"].stop

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    print(f"train length range: [{float(train['length_m'].min())*1000:.2f}, {hi*1000:.2f}] mm "
          f"-> {N_BINS} bins over [0,{hi*1000:.2f}] mm")

    bins_train = bin_index(train["length_m"], bin_edges)
    bins_test = bin_index(test["length_m"], bin_edges)

    results = {"bin_edges_mm": (bin_edges * 1000).tolist(), "n_bins": N_BINS,
               "min_rows_per_bin": MIN_ROWS_PER_BIN, "lam": {}}

    for lam in LAMS:
        print(f"\n=== lam={lam} ===")
        # switched fit
        A_sw, counts = fit_per_action_operators(train["phi_t"], train["phi_t1"], bins_train,
                                                  N_BINS, lam=lam)
        # bins with < MIN_ROWS_PER_BIN in TRAIN fall back to identity (persistence)
        for b in range(N_BINS):
            if int(counts[b]) < MIN_ROWS_PER_BIN:
                A_sw[b] = torch.eye(D)
        # global (unswitched) fit: one bin covering everything
        bins_all0 = torch.zeros_like(bins_train)
        A_gl, _ = fit_per_action_operators(train["phi_t"], train["phi_t1"], bins_all0, 1, lam=lam)

        pred_sw = apply_operators(A_sw, test["phi_t"], bins_test)
        pred_gl = apply_operators(A_gl, test["phi_t"], torch.zeros_like(bins_test))
        persist = test["phi_t"]
        truth = test["phi_t1"]

        acc_sw, err_sw, err_persist = descriptor_accuracy(pred_sw, truth, persist)
        acc_gl, err_gl, _ = descriptor_accuracy(pred_gl, truth, persist)

        per_bin = {}
        for b in range(N_BINS):
            m = bins_test == b
            nb = int(m.sum())
            if nb == 0:
                per_bin[b] = {"n_test": 0}
                continue
            a_sw, _, _ = descriptor_accuracy(pred_sw[m], truth[m], persist[m])
            a_gl, _, _ = descriptor_accuracy(pred_gl[m], truth[m], persist[m])
            per_bin[b] = {
                "n_train": int(counts[b]), "n_test": nb,
                "descriptor_accuracy_switched": a_sw,
                "descriptor_accuracy_global": a_gl,
                "lo_mm": float(bin_edges[b]) * 1000, "hi_mm": float(bin_edges[b + 1]) * 1000,
            }

        block_sw = block_rms(err_sw, slices)
        block_persist = block_rms(err_persist, slices)
        block_acc = {k: 1.0 - block_sw[k] / max(block_persist[k], 1e-12) for k in block_sw}

        print(f"overall descriptor_accuracy: switched={acc_sw:.4f} global={acc_gl:.4f}")
        print("per-bin (n_train/n_test, switched, global):")
        for b, row in per_bin.items():
            if row["n_test"] == 0:
                continue
            print(f"  bin{b} [{row['lo_mm']:.1f},{row['hi_mm']:.1f}]mm "
                  f"n_train={row['n_train']:5d} n_test={row['n_test']:5d} "
                  f"switched={row['descriptor_accuracy_switched']:.4f} "
                  f"global={row['descriptor_accuracy_global']:.4f}")
        print("by descriptor block (switched):", {k: round(v, 4) for k, v in block_acc.items()})

        results["lam"][str(lam)] = {
            "overall_descriptor_accuracy_switched": acc_sw,
            "overall_descriptor_accuracy_global": acc_gl,
            "per_bin": per_bin,
            "block_descriptor_accuracy_switched": block_acc,
            "train_bin_counts": counts.tolist(),
        }

    with open("experiments/temp/dmdc-lenbins/results_variantA.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/temp/dmdc-lenbins/results_variantA.json")


if __name__ == "__main__":
    main()
