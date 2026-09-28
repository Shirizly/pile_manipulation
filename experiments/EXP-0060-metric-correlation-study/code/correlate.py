"""EXP-0060: does any cheap offline IMAGE metric track `slateN` (the control metric that
actually matters) across model TYPES, and does it reward sharpness the way slateN does not?

Structure, per the design doc's section 4: a per-model row of {candidate image metrics,
slateN / slateN_tough}, correlated with Kendall tau. Built to APPEND, not rebuild: point
`--sources` at any number of `offline_eval*.json`-shaped files (same schema as
`EXP-0059-*/results/offline_eval_extended.json`: {model_name: {metric_key: value, ...}}) and
every model key across all of them becomes one row here. A new model (a retrieval variant, a
perturbed-simulator zoo member, ...) is added by scoring it into one of those JSON files with
`eval_extended.py`'s existing per-model keys and rerunning this script -- no code change needed
unless a genuinely new metric key is introduced.

Usage:
    python -u correlate.py --sources \
        ../../EXP-0059-retrieval-transition-model/results/offline_eval_extended.json

Current limitation (documented, not silently dropped): `slateN`/`slateN_tough` and every
diagnostic here are POPULATION means over DS-0009's 32 pools -- this script has one number per
model, not one per (model, pool), so a POOL-bootstrap CI on the correlation (what the brief asks
for) is not yet possible. That needs `eval_extended.py`'s pool loop to also persist its raw
per-pool-per-goal capture lists (`caps13`/`capsT`, currently averaged away before the JSON is
written) -- a small, well-localised change, deferred here because it requires a rerun of every
model (~4-8 min GPU/CPU) and the brief asked this record not to compete with the DS-0011/DS-0012
collection jobs for the machine. Until that lands, the Kendall tau below is taken ACROSS MODELS
(n = number of models scored so far), which is a legitimate but low-power test (this project's
own `paired_stats.required_n` guidance applies: at n=6, no correlation can be shown significant
at a conventional threshold) -- report it as suggestive, not conclusive, until n grows or the
pool-level version lands.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

# candidate offline metrics to test against slateN_tough (the headline per the coordinator's
# framing: does ranking/multi-step quality come from information, not blur -- so the CONTROL
# metric leads and these are all read as candidate proxies for it, never as verdicts themselves)
CANDIDATE_METRICS = [
    "accuracy_1",
    "accuracy_1_blur1",
    "accuracy_1_blur2",
    "occ_emd_swept",        # lower = better; sign-flipped before correlating (see below)
    "mass_in_goal_mae",     # lower = better; sign-flipped
    "mass_in_goal_mae_tough",  # lower = better; sign-flipped
]
TARGET = "slateN_tough"
TARGET_SECONDARY = "slateN"

# metrics where LOWER is better (a distance/error) get sign-flipped so every correlation in the
# table reads the same way: positive tau/rho = "tracks slateN correctly"
LOWER_IS_BETTER = {"occ_emd_swept", "mass_in_goal_mae", "mass_in_goal_mae_tough"}


def load_rows(paths):
    rows = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for model, r in d.items():
            if model in rows:
                print(f"WARNING: model '{model}' appears in more than one source file; "
                     f"keeping the first occurrence ({p} ignored for it)")
                continue
            rows[model] = r
    return rows


def build_table(rows):
    models = sorted(rows)
    table = {"model": models, TARGET: [], TARGET_SECONDARY: []}
    for m in models:
        table[TARGET].append(rows[m].get(TARGET, float("nan")))
        table[TARGET_SECONDARY].append(rows[m].get(TARGET_SECONDARY, float("nan")))
    for k in CANDIDATE_METRICS:
        vals = []
        for m in models:
            v = rows[m].get(k, float("nan"))
            if k in LOWER_IS_BETTER and v == v:  # not nan
                v = -v
            vals.append(v)
        table[k] = vals
    return table


def correlations(table):
    y = np.array(table[TARGET], dtype=float)
    out = {}
    for k in CANDIDATE_METRICS:
        x = np.array(table[k], dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        n = int(mask.sum())
        if n < 3:
            out[k] = dict(n=n, kendall_tau=None, kendall_p=None, spearman_r=None, spearman_p=None)
            continue
        tau, ptau = kendalltau(x[mask], y[mask])
        rho, prho = spearmanr(x[mask], y[mask])
        out[k] = dict(n=n, kendall_tau=float(tau), kendall_p=float(ptau),
                      spearman_r=float(rho), spearman_p=float(prho))
    return out


# --------------------------------------------------------------------------
# Pool-level analysis (added 2026-09-28, once eval_extended.py started
# persisting per-pool raw {pool, vt, vp} arrays to a sibling "*_raw.json" --
# see EXP-0059/LOG.md's 2026-09-28 entry). Two things this unlocks that the
# cross-model table above cannot:
#   (a) a POOL-BOOTSTRAP CI on slateN_tough per model (resample the 32 pools
#       with replacement, recompute the capture mean each time) -- answers
#       "how much would this model's slateN_tough move under a different
#       32-pool sample", independent of how many OTHER models exist;
#   (b) within-pool Spearman(vp, vt) -- does the model's PREDICTED dv ordering
#       correlate with the TRUE dv ordering inside one pool's 64 candidates,
#       averaged over pools and goals. This is closer to "ranking quality" than
#       slateN itself (which only reads off the single top-1 pick).
# --------------------------------------------------------------------------

def _capture(vt: np.ndarray, vp: np.ndarray):
    den = vt.mean() - vt.min()
    return float((vt.mean() - vt[vp.argmin()]) / den) if den > 1e-9 else None


def pool_bootstrap_ci(raw_goal_dict, n_boot=1000, seed=0, alpha=0.05):
    """raw_goal_dict: {goal: [{"pool":int,"vt":[...],"vp":[...]}, ...]} (one model's
    slateN_tough_raw, say). -> (mean, lo, hi) percentile CI on the goal-averaged capture,
    bootstrapping over POOLS (shared resample across goals, since every goal is scored on the
    SAME 32 pools -- resampling per-goal independently would overstate cross-goal averaging)."""
    goals = sorted(raw_goal_dict)
    # align every goal's rows by pool id (defensive: DS-0009/0011 write goals in the same
    # pool order today, but this does not assume it)
    pool_ids = sorted({row["pool"] for row in raw_goal_dict[goals[0]]})
    n_pools = len(pool_ids)
    CAP = np.full((len(goals), n_pools), np.nan)
    for gi, g in enumerate(goals):
        by_pool = {row["pool"]: row for row in raw_goal_dict[g]}
        for pi, pid in enumerate(pool_ids):
            row = by_pool.get(pid)
            if row is None:
                continue
            c = _capture(np.array(row["vt"]), np.array(row["vp"]))
            if c is not None:
                CAP[gi, pi] = c
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pools, size=(n_boot, n_pools))
    boots = np.nanmean(np.nanmean(CAP[:, idx], axis=2), axis=0)  # (n_boot,)
    point = float(np.nanmean(np.nanmean(CAP, axis=1)))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def within_pool_spearman(raw_goal_dict):
    """Mean Spearman(vp, vt) over every (goal, pool) in this model's raw dict."""
    rhos = []
    for g, rows in raw_goal_dict.items():
        for row in rows:
            vt, vp = np.array(row["vt"]), np.array(row["vp"])
            if np.std(vt) < 1e-12 or np.std(vp) < 1e-12:
                continue  # degenerate pool (no true or predicted spread to rank)
            rho, _ = spearmanr(vt, vp)
            if rho == rho:  # not nan
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


def pool_level_table(raw_paths):
    """{model: {slateN_tough_boot: (mean,lo,hi), within_pool_spearman_tough: rho}}"""
    out = {}
    for p in raw_paths:
        d = json.loads(Path(p).read_text())
        for model, r in d.items():
            if "slateN_tough_raw" not in r:
                continue
            if model in out:
                continue
            mean, lo, hi = pool_bootstrap_ci(r["slateN_tough_raw"])
            rho = within_pool_spearman(r["slateN_tough_raw"])
            out[model] = dict(slateN_tough_boot_mean=mean, slateN_tough_boot_lo=lo,
                              slateN_tough_boot_hi=hi, within_pool_spearman_tough=rho)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--raw-sources", nargs="*", default=[],
                   help="sibling *_raw.json files (per-pool {pool,vt,vp} arrays) -- enables "
                        "pool-bootstrap CI + within-pool Spearman(vp,vt); omit to skip both")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "correlations.json"))
    a = ap.parse_args()

    rows = load_rows(a.sources)
    table = build_table(rows)
    corr = correlations(table)
    pool_level = pool_level_table(a.raw_sources) if a.raw_sources else {}

    out = dict(
        n_models=len(table["model"]),
        models=table["model"],
        target=TARGET,
        lower_is_better_flipped=sorted(LOWER_IS_BETTER),
        table=table,
        correlations=corr,
        pool_level=pool_level,
        caveat="Kendall tau / Spearman rho computed ACROSS MODELS (n=n_models). pool_level (if "
               "present) is the POOL-bootstrap CI + within-pool Spearman(vp,vt), a different, "
               "per-model quantity that does not need n_models>=3 to be meaningful. Sign-flipped "
               "so positive always means 'tracks slateN_tough correctly'.",
    )
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(out, indent=1))
    import os
    os.replace(tmp, out_path)

    print(f"n_models = {out['n_models']}: {table['model']}")
    print(f"{'metric':24s} {'n':>3s} {'tau':>7s} {'p(tau)':>7s} {'rho':>7s} {'p(rho)':>7s}")
    for k, c in corr.items():
        if c["kendall_tau"] is None:
            print(f"{k:24s} {c['n']:>3d}   (too few non-nan values)")
            continue
        print(f"{k:24s} {c['n']:>3d} {c['kendall_tau']:+7.3f} {c['kendall_p']:7.3f} "
             f"{c['spearman_r']:+7.3f} {c['spearman_p']:7.3f}")
    if pool_level:
        print(f"\n{'model':36s} {'slateN_tough (95% pool-boot CI)':>34s}  {'within-pool rho(vp,vt)':>22s}")
        for m, d in pool_level.items():
            rho_s = f"{d['within_pool_spearman_tough']:+.3f}" if d['within_pool_spearman_tough'] is not None \
                else "n/a (degenerate: predictor has zero predicted-dv spread every pool)"
            print(f"{m:36s} {d['slateN_tough_boot_mean']:+.3f} [{d['slateN_tough_boot_lo']:+.3f}, "
                 f"{d['slateN_tough_boot_hi']:+.3f}]{'':>6s} {rho_s}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
