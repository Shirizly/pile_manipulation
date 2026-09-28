"""EXP-0026 A1: paired re-analysis of EXP-0023's 60 (state, arm) cells.

Reads EXP-0023's results/metrics.json (no re-simulation). Every quantity is
turned into HIGHER = BETTER before analysis: `gradient_gain` and `pool_escape`
already are; `dv_rank` / `dv_grad` are lyapunov COSTS and are negated via
`goals.improvement(dv, 0, higher_is_better=False)`.
Known-number check first: `variance_components(gradient_gain).residual_sd`
must reproduce EXP-0023's own 0.01826.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import improvement, higher_is_better_for
from Baselines.common.paired_stats import (friedman, paired_comparison, power_table,
                                           rank_stability, variance_components)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = glob.glob(str(REPO / "experiments/EXP-0023-*/results/metrics.json"))[0]
    d = json.load(open(src))
    arms, slates = d["arms"], d["slates"]
    hib = higher_is_better_for("lyapunov")

    def mat(key, to_gain):
        X = np.array([[next(r[key] for r in d["rows"] if r["arm"] == a and r["slate"] == s)
                       for s in slates] for a in arms], dtype=float)
        return improvement(X, 0.0, higher_is_better=hib) if to_gain else X

    quantities = {
        "gradient_gain": (mat("gradient_gain", False), [0.005, 0.01, 0.02]),
        "pool_escape":   (mat("pool_escape", False),   [0.005, 0.01, 0.02]),
        "neg_dv_grad":   (mat("dv_grad", True),        [0.005, 0.01, 0.02]),
        "neg_dv_rank":   (mat("dv_rank", True),        [0.005, 0.01, 0.02]),
    }
    vc = variance_components(quantities["gradient_gain"][0])
    # EXP-0023 took np.std(ddof=1) over all M*S residual cells (df = M*S-1 = 59);
    # the two-way residual df is (M-1)(S-1) = 45. Same sum of squares, so the
    # two must agree after rescaling by sqrt(59/45).
    M, S = len(arms), len(slates)
    theirs = d["power"]["residual_sd"] * np.sqrt((M * S - 1) / ((M - 1) * (S - 1)))
    assert abs(vc["residual_sd"] - theirs) < 1e-6, (vc, d["power"])
    print(f"known-number check: residual_sd {vc['residual_sd']:.5f} == EXP-0023's "
          f"{d['power']['residual_sd']:.5f} rescaled to df=(M-1)(S-1) ({theirs:.5f})")

    out = {"source": "EXP-0023 results/metrics.json", "arms": arms, "slates": slates,
           "quantities": {}}
    for q, (X, deltas) in quantities.items():
        pairs = paired_comparison(X, arms)
        res = dict(means={a: float(X[k].mean()) for k, a in enumerate(arms)},
                   variance_components=variance_components(X), friedman=friedman(X),
                   pairs=pairs, power=power_table(pairs, deltas),
                   rank_stability=rank_stability(X, arms),
                   n_pairs_resolved_ci=sum(p["resolved_ci"] for p in pairs),
                   n_pairs_holm_05=sum(p["p_holm"] < 0.05 for p in pairs))
        out["quantities"][q] = res
        vcq, fr, pw = res["variance_components"], res["friedman"], res["power"]
        print(f"\n== {q} ==  (higher = better)")
        print("  means: " + "  ".join(f"{a}={m:+.4f}" for a, m in res["means"].items()))
        print(f"  sd: model-means {vcq['model_sd_of_means']:.4f}  state-means "
              f"{vcq['state_sd_of_means']:.4f}  residual {vcq['residual_sd']:.4f}  "
              f"ANOVA p={vcq['anova_p_model']:.3g}  Friedman p={fr['p']:.3g} W={fr['kendall_w']:.2f}")
        print(f"  pairs resolved: CI {res['n_pairs_resolved_ci']}/15, Holm<0.05 "
              f"{res['n_pairs_holm_05']}/15; median sd_diff {pw['median_sd_diff']:.4f}")
        print("  required states: " + "  ".join(
            f"delta={k}: {v['alpha_05']} (Bonf {v['bonferroni']})" for k, v in pw["required_n"].items()))
        for p in sorted(pairs, key=lambda r: r["p_signflip"])[:4]:
            print(f"   {p['a']:>22s} - {p['b']:<22s} {p['mean_diff']:+.4f} "
                  f"[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] p={p['p_signflip']:.3f} "
                  f"holm={p['p_holm']:.3f} wins {p['wins_a']}-{p['wins_b']}")
        print("  P(first): " + "  ".join(f"{a}={v['p_first']:.2f}" for a, v in res["rank_stability"].items()))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
