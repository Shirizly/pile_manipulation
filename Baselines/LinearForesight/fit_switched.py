"""Baselines/LinearForesight/fit_switched.py -- fit Suh & Tedrake 2020's
switched-linear operator, ONE PER PUSH-LENGTH BIN, on the overnight_randlen
corpus (`Genesis/data/overnight_randlen`).

Why this script exists, distinct from the repo-root `fit_linear_foresight.py`
--------------------------------------------------------------------------
`fit_linear_foresight.py` fits and falsifies a SINGLE operator on datasets
collected at ONE fixed push length (`--perpendicular-pushes --push-length`,
see `docs/linear_visual_foresight_baseline.md` Step 6/§7). `overnight_randlen`
is the opposite: push length is randomised per transition
(`Genesis/data/overnight_randlen/DATASET.yaml`: realised length ranges
~0.0005-0.08 m, no fixed value), so a single operator over the whole corpus
-- which is exactly what `Baselines/common/eval_baseline.py`/
`eval_randlen_indist.py`'s "linear" reference row already does -- is not the
paper's switched-linear model at all, just its degenerate 1-bin case. This
script is the actual switched design (their §3.1, 5 length bins) applied to
this corpus: split transitions into `--n-bins` (default 6) equal-width bins
over the OBSERVED push-length range, fit one operator per bin, and report
whether switching beats the single global operator on held-out data.

Bin edges: [0, max(push_length_m)] split into `n_bins` equal-width bins.
Zero (not the observed minimum) is used as the low edge because push length
is a non-negative physical quantity and a fixed floor keeps bin edges
reproducible across train subsamples; the observed minimum is a few tenths
of a millimetre away from zero (`min_push_length_m: 0.0001` already filters
near-zero pushes) so this makes a negligible difference to bin membership.

Data loading goes through `Baselines.common.randlen_data.load_randlen_cell`
(registry-based, N-agnostic) rather than `Baselines.common.data.load_cell`,
so all 5 overnight_randlen groups (mixed_n20, piled_n20, scattered_n20,
piled_n50, scattered_n50) are used, not just the n20 subset `CellData`'s
fixed (n,20,7) states allocation can read (user decision, 2026-09-10).
Called with `need_step_idx=False`: fitting only ever needs `occ0`/`occ1`/
`actions`, never the step-0-candidate slate structure, and the pooled
`_all.yaml` configs used here are big enough (192/21 files) that a
`min_push_length_m` filter dropping even one row somewhere makes
`load_randlen_cell`'s default step-tagging assertion fire (measured: 5
rows short of 24576 on the train split) -- a real fact about this corpus,
not a bug, and irrelevant to a fit that never looks at step_idx.

No new fit/apply math: per-bin operators reuse `fit_linear_foresight.
fit_operator`/`fit_operator_nonneg` (their eq. 8/9) and are applied via
`Baselines.LinearForesight.model.predict_switched`, which is the same
`predict_world` warp/matvec/unwarp/blend pipeline dispatched per bin.

Usage
-----
    # res=64 (this repo's convention, matches the existing single-operator
    # reference) -- writes runs/operators_res64.pt
    PYTHONPATH=. python Baselines/LinearForesight/fit_switched.py --res 64

    # res=32 (the paper's own resolution) -- writes runs/operators_res32.pt,
    # for the resolution ablation docs/linear_visual_foresight_baseline.md
    # §8 calls for
    PYTHONPATH=. python Baselines/LinearForesight/fit_switched.py --res 32
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, fit_operator_nonneg,
    metrics, predict_world, swept_region_mask,
)
from scripts.probes.exp0009_rerun import predict_meandelta
from utils import git_provenance

from Baselines.common.randlen_data import load_randlen_cell
from Baselines.LinearForesight.model import bin_index, predict_switched, push_length_m

TRAIN_CFG = "configs/dataset/genesis_overnight_randlen_train_all.yaml"
TEST_CFG = "configs/dataset/genesis_overnight_randlen_test_all.yaml"
MIN_ROWS_PER_BIN = 50  # below this a bin's fit is not meaningfully better than
                       # guessing; falls back to the identity (persistence)
                       # operator for that bin rather than fitting on noise.


def _fit_one(Y0: torch.Tensor, Y1: torch.Tensor, constraint: str, ridge: float) -> torch.Tensor:
    if constraint == "nonneg":
        return fit_operator_nonneg(Y0, Y1, ridge=ridge, toward_identity=True)
    return fit_operator(Y0, Y1, ridge=ridge, toward_identity=True)


def fit_bins(occ0, occ1, s_px, e_px, lengths_m, bin_edges, res, crop, constraint, ridge, device):
    n_bins = len(bin_edges) - 1
    bins = bin_index(lengths_m, bin_edges)
    operators, counts = [], []
    for b in range(n_bins):
        m = bins == b
        n_b = int(m.sum())
        counts.append(n_b)
        lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
        if n_b < MIN_ROWS_PER_BIN:
            print(f"  bin {b} [{lo * 1000:.1f},{hi * 1000:.1f}) mm: {n_b} rows "
                  f"(< {MIN_ROWS_PER_BIN}), falling back to identity (persistence)")
            R = res
            operators.append(torch.eye(R * R))
            continue
        Y0 = canonicalise(occ0[m].to(device), s_px[m].to(device), e_px[m].to(device),
                           res, crop).reshape(n_b, -1).T
        Y1 = canonicalise(occ1[m].to(device), s_px[m].to(device), e_px[m].to(device),
                           res, crop).reshape(n_b, -1).T
        t0 = time.time()
        A = _fit_one(Y0, Y1, constraint, ridge).cpu()
        print(f"  bin {b} [{lo * 1000:.1f},{hi * 1000:.1f}) mm: {n_b} rows, "
              f"fit {constraint} in {time.time() - t0:.1f}s")
        operators.append(A)
    return operators, counts


def evaluate(operators, bin_edges, res, crop, train_bmd, train_A_single,
             test_cell, H, W):
    lengths_te = push_length_m(test_cell.actions)
    s_te, e_te = actions_to_pixels(test_cell.actions, test_cell.workspace_min,
                                    test_cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_te, e_te, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    preds = {
        "persistence": test_cell.occ0,
        "mean-delta": predict_meandelta(train_bmd, test_cell.occ0, s_te, e_te, res, (H, W), crop),
        "linear-single": predict_world(train_A_single, test_cell.occ0, s_te, e_te, res, (H, W), crop),
        "linear-switched": predict_switched(bin_edges, operators, test_cell.occ0, s_te, e_te,
                                             lengths_te, res, (H, W), crop),
    }

    results = {name: metrics(pred, test_cell.occ1, test_cell.occ0, region=region)
               for name, pred in preds.items()}

    # Per-bin breakdown on the SAME test set, so "switching helps" can be
    # read off directly instead of only in aggregate.
    bins_te = bin_index(lengths_te, bin_edges)
    per_bin = {}
    for b in range(len(operators)):
        m = bins_te == b
        if int(m.sum()) == 0:
            continue
        per_bin[b] = {name: metrics(pred[m], test_cell.occ1[m], test_cell.occ0[m],
                                     region=region[m])["accuracy"]
                      for name, pred in preds.items()}
    return results, per_bin


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cfg", default=TRAIN_CFG)
    ap.add_argument("--test-cfg", default=TEST_CFG)
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--res", type=int, default=64,
                    help="canonical-frame fit resolution. 64 (default) matches "
                         "this repo's established single-operator 'linear' "
                         "reference (Baselines/common/eval_baseline.py's "
                         "R=64) for direct comparability; the paper itself "
                         "used 32.")
    ap.add_argument("--crop", type=float, default=1.0)
    ap.add_argument("--constraint", choices=["ridge", "nonneg"], default="ridge",
                     help="ridge (default) matches the existing single-operator "
                          "'linear' reference's fit (ridge=1.0 toward identity); "
                          "nonneg is the paper's own best-performing variant "
                          "(their Fig. 7) but is not what today's single-operator "
                          "reference uses, so 'ridge' is the fairer switched-vs-"
                          "single comparison by default.")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                     help="default: Baselines/LinearForesight/runs/"
                          "operators_res<R>.pt, derived from --res so a "
                          "resolution ablation (e.g. --res 32) never silently "
                          "overwrites the --res 64 bundle.")
    ap.add_argument("--max-samples", type=int, default=None,
                     help="subsample the TRAIN split for a quick smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"Baselines/LinearForesight/runs/operators_res{args.res}.pt"

    print(f"=== switched-linear foresight: {args.n_bins} push-length bins "
          f"@ {args.res}x{args.res}, constraint={args.constraint} ===")

    t0 = time.time()
    data = load_randlen_cell(args.train_cfg, "train", tag="train", need_step_idx=False)
    print(f"loaded {data.occ0.shape[0]} train transitions from "
          f"{data.run_idx.unique().numel()} files, grid {data.H}x{data.W}, "
          f"{time.time() - t0:.1f}s")

    if args.max_samples and args.max_samples < data.occ0.shape[0]:
        idx = torch.randperm(data.occ0.shape[0])[:args.max_samples]
        for f in ("occ0", "occ1", "actions", "run_idx"):
            setattr(data, f, getattr(data, f)[idx])
        print(f"  subsampled to {args.max_samples} rows (--max-samples)")

    H, W = data.H, data.W
    lengths_m = push_length_m(data.actions)
    hi = float(lengths_m.max())
    bin_edges = torch.linspace(0.0, hi, args.n_bins + 1)
    print(f"push length range: min {float(lengths_m.min()) * 1000:.2f} mm, "
          f"max {hi * 1000:.2f} mm -> {args.n_bins} equal-width bins over "
          f"[0, {hi * 1000:.2f}] mm")

    s_px, e_px = actions_to_pixels(data.actions, data.workspace_min, data.workspace_max, (H, W))

    print(f"\nfitting per-bin operators ({args.constraint}, ridge={args.ridge}):")
    operators, counts = fit_bins(data.occ0, data.occ1, s_px, e_px, lengths_m, bin_edges,
                                  args.res, args.crop, args.constraint, args.ridge, args.device)

    # Single global reference operator, fit on the SAME train data, for the
    # switched-vs-single comparison (identical fit recipe modulo the
    # switching itself, ridge=1.0 toward identity -- the recipe today's
    # single 'linear' reference row already uses).
    Y0_all = canonicalise(data.occ0.to(args.device), s_px.to(args.device),
                           e_px.to(args.device), args.res, args.crop
                           ).reshape(data.occ0.shape[0], -1).T
    Y1_all = canonicalise(data.occ1.to(args.device), s_px.to(args.device),
                           e_px.to(args.device), args.res, args.crop
                           ).reshape(data.occ0.shape[0], -1).T
    A_single = fit_operator(Y0_all, Y1_all, ridge=1.0, toward_identity=True).cpu()
    bmd = (Y1_all - Y0_all).mean(dim=1).cpu()

    # `single_operator`/`mean_delta` are bundled alongside the switched
    # per-bin operators (not a separate file/run) SPECIFICALLY so the
    # switched-vs-single comparison is always apples-to-apples: both are fit
    # on the identical loaded train tensors, in the same process, at the
    # same resolution -- never a stale/previously-saved single operator.
    # `Baselines/LinearForesight/predictor.py`'s
    # `SingleLinearForesightPredictor` reads `single_operator` straight from
    # this same bundle for exactly this reason.
    ckpt = {
        "operators": operators, "bin_edges": bin_edges, "counts": counts,
        "single_operator": A_single, "mean_delta": bmd,
        "res": args.res, "crop": args.crop, "constraint": args.constraint,
        "ridge": args.ridge, "n_bins": args.n_bins,
        "train_cfg": args.train_cfg, "provenance": git_provenance(),
    }
    torch.save(ckpt, args.out)
    print(f"\nwrote {args.out}")

    print(f"\nloading held-out test split ({args.test_cfg}) ...")
    test_cell = load_randlen_cell(args.test_cfg, "train", tag="test", need_step_idx=False)
    print(f"loaded {test_cell.occ0.shape[0]} test transitions from "
          f"{test_cell.run_idx.unique().numel()} files")

    results, per_bin = evaluate(operators, bin_edges, args.res, args.crop, bmd, A_single,
                                 test_cell, H, W)

    print(f"\n=== held-out one-step error, swept region only "
          f"({test_cell.occ0.shape[0]} transitions) ===")
    hdr = f"{'model':20s} {'rms':>9s} {'l1/mass':>9s} {'accuracy':>10s}"
    print(hdr); print("-" * len(hdr))
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rms"]):
        print(f"{name:20s} {m['rms']:9.5f} {m['l1_per_mass']:9.4f} {m['accuracy']:10.4f}")

    print(f"\n=== per-bin accuracy on held-out test (0 = persistence) ===")
    names = list(next(iter(per_bin.values())).keys()) if per_bin else []
    print(f"{'bin (mm)':>16s} " + "".join(f"{n[:14]:>15s}" for n in names))
    for b, row in per_bin.items():
        lo, hi_b = float(bin_edges[b]) * 1000, float(bin_edges[b + 1]) * 1000
        label = f"[{lo:.1f},{hi_b:.1f})"
        print(f"{label:>16s} " + "".join(f"{row[n]:15.4f}" for n in names))

    acc_path = args.out.rsplit(".", 1)[0] + "_accuracy.json"
    with open(acc_path, "w") as f:
        json.dump({
            "provenance": ckpt["provenance"], "train_cfg": args.train_cfg,
            "test_cfg": args.test_cfg, "n_bins": args.n_bins, "res": args.res,
            "constraint": args.constraint, "counts_per_bin": counts,
            "bin_edges_mm": [float(x) * 1000 for x in bin_edges.tolist()],
            "accuracy": {k: v["accuracy"] for k, v in results.items()},
            "per_bin_accuracy": {str(b): row for b, row in per_bin.items()},
        }, f, indent=2)
    print(f"\nwrote {acc_path}")


if __name__ == "__main__":
    main()
