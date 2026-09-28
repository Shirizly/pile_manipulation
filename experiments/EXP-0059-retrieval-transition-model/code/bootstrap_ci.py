"""Coordinator follow-up (2026-09-28): pool-bootstrap 95% CIs on the data-scaling
`slateN_tough` values and the multi-step terminal values, plus PAIRED-delta CIs for
25k-vs-98k (data scaling) and linear64-vs-narrow-NFD at 3 steps (multi-step).
Reuses EXP-0060's `correlate.py::pool_bootstrap_ci`/`_capture` unchanged (same
formula, same resampling convention: shared pool resample across goals).

Usage:
    python -u bootstrap_ci.py
"""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/EXP-0060-metric-correlation-study/code"))
import correlate as CO  # noqa: E402  (pool_bootstrap_ci, _capture)

EXP0059 = Path(__file__).resolve().parents[1]
DS_RAW = EXP0059 / "results" / "data_scaling_pools_raw.json"
MS_RAW = EXP0059 / "results" / "multistep_eval_ci_raw.json"
OUT = EXP0059 / "results" / "bootstrap_ci.json"


def _align_capture_matrix(raw_goal_dict):
    """{goal: [{"pool","vt","vp"}]} -> (n_goals, n_pools) capture matrix, pool-id
    ordered, NaN where degenerate/missing -- the same alignment `pool_bootstrap_ci`
    builds internally, exposed here so a PAIRED delta can share one resample."""
    goals = sorted(raw_goal_dict)
    pool_ids = sorted({row["pool"] for row in raw_goal_dict[goals[0]]})
    CAP = np.full((len(goals), len(pool_ids)), np.nan)
    for gi, g in enumerate(goals):
        by_pool = {row["pool"]: row for row in raw_goal_dict[g]}
        for pi, pid in enumerate(pool_ids):
            row = by_pool.get(pid)
            if row is None:
                continue
            c = CO._capture(np.array(row["vt"]), np.array(row["vp"]))
            if c is not None:
                CAP[gi, pi] = c
    return CAP, pool_ids


def paired_delta_ci(raw_a, raw_b, n_boot=2000, seed=0, alpha=0.05):
    """raw_a, raw_b: two models' {goal: [{"pool","vt","vp"}]} SCORED ON THE SAME
    POOLS -- resamples pool indices ONCE per draw (shared across both models),
    so the delta captures the PAIRED comparison, not an independent-samples one."""
    CAP_a, pids_a = _align_capture_matrix(raw_a)
    CAP_b, pids_b = _align_capture_matrix(raw_b)
    assert pids_a == pids_b, "the two models must be scored on the same pool id set"
    n_pools = CAP_a.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pools, size=(n_boot, n_pools))
    boot_a = np.nanmean(np.nanmean(CAP_a[:, idx], axis=2), axis=0)
    boot_b = np.nanmean(np.nanmean(CAP_b[:, idx], axis=2), axis=0)
    delta = boot_b - boot_a
    point = float(np.nanmean(np.nanmean(CAP_b, axis=1)) - np.nanmean(np.nanmean(CAP_a, axis=1)))
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac_gt0 = float((delta > 0).mean())
    return point, float(lo), float(hi), frac_gt0


def main():
    out = {}

    if DS_RAW.exists():
        ds = json.loads(DS_RAW.read_text())
        out["data_scaling"] = {}
        for name, d in ds.items():
            mean, lo, hi = CO.pool_bootstrap_ci(d["slateN_tough_raw"])
            out["data_scaling"][name] = dict(point=d["point"], boot_mean=mean, ci95=[lo, hi])
            print(f"data_scaling {name:16s} slateN_tough={d['point']:.4f}  boot CI95=[{lo:.4f},{hi:.4f}]")
        if "25k" in ds and "98k_all" in ds:
            point, lo, hi, frac_gt0 = paired_delta_ci(ds["25k"]["slateN_tough_raw"], ds["98k_all"]["slateN_tough_raw"])
            out["data_scaling"]["paired_delta_98k_minus_25k"] = dict(point=point, ci95=[lo, hi], frac_boot_gt0=frac_gt0)
            print(f"paired delta (98k - 25k): {point:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
                 f"frac(boot>0)={frac_gt0:.3f}")
    else:
        print("data_scaling_pools_raw.json not found yet -- skipping data-scaling CIs")

    if MS_RAW.exists():
        ms = json.loads(MS_RAW.read_text())
        out["multistep_terminal"] = {}
        for name, d in ms.items():
            if "slateN_tough_terminal_raw" not in d:
                continue
            mean, lo, hi = CO.pool_bootstrap_ci(d["slateN_tough_terminal_raw"])
            out["multistep_terminal"][name] = dict(boot_mean=mean, ci95=[lo, hi])
            print(f"multistep terminal {name:36s} boot_mean={mean:.4f}  CI95=[{lo:.4f},{hi:.4f}]")
        if "linear_narrow_l20_res64" in ms and "nfd_3ch_narrow_l20" in ms:
            point, lo, hi, frac_gt0 = paired_delta_ci(
                ms["nfd_3ch_narrow_l20"]["slateN_tough_terminal_raw"],
                ms["linear_narrow_l20_res64"]["slateN_tough_terminal_raw"])
            out["multistep_terminal"]["paired_delta_linear64_minus_narrowNFD"] = dict(
                point=point, ci95=[lo, hi], frac_boot_gt0=frac_gt0)
            print(f"paired delta (linear64 - narrowNFD, terminal): {point:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
                 f"frac(boot>0)={frac_gt0:.3f}")
    else:
        print("multistep_eval_ci_raw.json not found yet -- skipping multi-step CIs")

    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
