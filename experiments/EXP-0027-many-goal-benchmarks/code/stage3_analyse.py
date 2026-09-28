"""EXP-0027 item 2, stage 3: score every simulated action under every goal
(from banked final positions) and run the paired analysis.

Per (state s, goal g): pool true dv over the 100 pool actions (same path), then
per arm
  rank_capture = (mean_pool - dv_rank) / (mean_pool - best_pool)   (= slateN of the pick)
  grad_capture = (mean_pool - dv_grad) / (mean_pool - best_pool)   (> 1: beat the pool)
lyapunov is a COST, so "best" is the min (goals.best_of / improvement).
Cells whose pool range < MIN_RANGE are dropped and counted.

C4: ICC over goals of per-(s, g) arm-pair differences in grad_capture, and the
    sd shrink of the per-state paired difference from G=1 to G=12.
C5: Holm-significant arm pairs on grad_capture (goal-averaged per state).
Also: optimiser noise (this run's GD vs EXP-0023's GD, same model/state/goal,
slates 0-9, corner), ranking-vs-gradient agreement, bound-hit rates.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import best_of, higher_is_better_for, improvement
from Baselines.common.paired_stats import (friedman, paired_comparison, rank_stability,
                                           required_n, variance_components)
from simple_mpc.adapters import occ_from_particles
from simple_mpc.gt_bank import GroundTruthBank

MIN_RANGE = 1e-4
HIB = higher_is_better_for("lyapunov")


def lyap_fields(occ, D, eps=1e-6):
    """occ (B,H,W), D (G,H,W) -> (B, G) lyapunov values."""
    f = occ.reshape(occ.shape[0], -1)
    return (f @ D.reshape(D.shape[0], -1).T) / f.sum(1, keepdim=True).clamp_min(eps)


def capture(dv, pool):
    """(dv - pool mean) improvement over (best - mean), sense-safe."""
    return float(improvement(dv, pool.mean(), higher_is_better=HIB) /
                 improvement(best_of(pool, higher_is_better=HIB), pool.mean(), higher_is_better=HIB))


def icc_and_shrink(C):
    """C: (A, S, G). Median over arm pairs of the goal-ICC of the paired
    difference, and median per-pair sd of the per-state difference at G=1
    (median over goals) and G=all."""
    iccs, sd1, sdG = [], [], []
    for i, j in itertools.combinations(range(C.shape[0]), 2):
        D = (C[i] - C[j]).T                                  # (G, S)
        keep = ~np.isnan(D).any(0)
        Dk = D[:, keep]
        G, S = Dk.shape
        msb = G * Dk.mean(0).var(ddof=1)
        msw = ((Dk - Dk.mean(0)) ** 2).sum() / (S * (G - 1))
        s2 = max((msb - msw) / G, 0.0)
        iccs.append(s2 / (s2 + msw))
        sd1.append(np.median([np.nanstd(D[g], ddof=1) for g in range(D.shape[0])]))
        sdG.append(np.nanstd(np.nanmean(D, 0), ddof=1))
    return float(np.median(iccs)), float(np.median(sd1)), float(np.median(sdG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1 = torch.load(a.stage1, map_location="cpu", weights_only=False)
    s2 = torch.load(a.stage2, map_location="cpu", weights_only=False)
    bank = GroundTruthBank(REPO / "datasets/DS-0004-ground-truth-bank/data")
    D = s1["goal_fields"].float()
    arms, goals, slates = list(s1["arms"]), s1["goals"], s2["slates"]
    A, S, G = len(arms), len(slates), len(goals)
    RC = np.full((A, S, G), np.nan); GC = np.full((A, S, G), np.nan)
    DVG = np.full((A, S, G), np.nan); DVR = np.full((A, S, G), np.nan)
    noise = []                                   # (arm, slate, cap_this, cap_exp0023)
    n_dropped = 0
    for si, s in enumerate(slates):
        rec = bank.record("oracle_rollout_full", s2["state_keys"][s])
        L = s2["labels"][s]
        st = rec["state"]
        v0 = lyap_fields(occ_from_particles(st[None, :, :3], "cpu"), D)[0]          # (G,)
        hit, fin = bank.lookup(st, L["actions"], "oracle_rollout_full")
        assert hit.all()
        dv = lyap_fields(occ_from_particles(fin[..., :3], "cpu"), D) - v0[None]      # (N, G)
        lab = L["labels"]
        pool = dv[[i for i, l in enumerate(lab) if l[0] == "pool"]]                  # (100, G)
        idx = {(l[0], l[1], l[2]): i for i, l in enumerate(lab) if l[0] != "pool"}
        for gi in range(G):
            p = pool[:, gi]
            if float(p.max() - p.min()) < MIN_RANGE:
                n_dropped += 1
                continue
            for ai, arm in enumerate(arms):
                dr, dg = dv[idx[("rank", arm, gi)], gi], dv[idx[("grad", arm, gi)], gi]
                DVR[ai, si, gi], DVG[ai, si, gi] = float(dr), float(dg)
                RC[ai, si, gi], GC[ai, si, gi] = capture(dr, p), capture(dg, p)
                if gi == 0 and ("grad23", arm, 0) in idx:
                    noise.append((arm, s, GC[ai, si, gi], capture(dv[idx[("grad23", arm, 0)], 0], p)))
    out = {"arms": arms, "goals": goals, "slates": slates, "n_cells_dropped": n_dropped,
           "min_range": MIN_RANGE}

    def analyse(X3, name):
        X = np.nanmean(X3, 2)                                  # goal-averaged (A, S)
        pairs = paired_comparison(X, arms)
        icc, sd1, sdG = icc_and_shrink(X3)
        r = dict(means={m: float(np.nanmean(X[k])) for k, m in enumerate(arms)},
                 per_goal_means={m: [float(np.nanmean(X3[k, :, g])) for g in range(G)]
                                 for k, m in enumerate(arms)},
                 friedman=friedman(X), variance_components=variance_components(X),
                 pairs=pairs, rank_stability=rank_stability(X, arms),
                 n_resolved_ci=sum(p["resolved_ci"] for p in pairs),
                 n_holm_05=sum(p["p_holm"] < .05 for p in pairs),
                 icc_goals=icc, median_sd_G1=sd1, median_sd_Gall=sdG, sd_shrink=sd1 / sdG,
                 req_states_d005={"G1": required_n(sd1, 0.05), "Gall": required_n(sdG, 0.05)},
                 req_states_d01={"G1": required_n(sd1, 0.10), "Gall": required_n(sdG, 0.10)})
        out[name] = r
        return r

    for name, X3 in (("grad_capture", GC), ("rank_capture", RC), ("gain_capture", GC - RC)):
        analyse(X3, name)
    # optimiser noise: two GD runs of the same (arm, state, corner), same path
    nz = np.array([(n[2], n[3]) for n in noise])
    diffs = nz[:, 0] - nz[:, 1]
    out["optimiser_noise"] = dict(
        n=len(noise), sd_single_run=float(diffs.std(ddof=1) / np.sqrt(2)),
        mean_abs_diff=float(np.abs(diffs).mean()), max_abs_diff=float(np.abs(diffs).max()),
        per_arm={arm: float(np.std([n[2] - n[3] for n in noise if n[0] == arm], ddof=1) / np.sqrt(2))
                 for arm in arms},
        note="grad_capture units; sd of ONE GD run's outcome from numerical perturbation alone")
    # ranking vs gradient: do the two abilities order the arms the same way?
    t, p = stats.kendalltau([out["rank_capture"]["means"][m] for m in arms],
                            [out["grad_capture"]["means"][m] for m in arms])
    out["rank_vs_grad_kendall"] = dict(tau=float(t), p=float(p))
    out["bound_hit"] = {arm: [float(x) for x in torch.stack(list(s1["arms"][arm]["bound_hit_frac"].values())).mean((0, 1))]
                        for arm in arms}
    out["frac_grad_beats_pool"] = {m: float(np.nanmean(GC[k] > 1)) for k, m in enumerate(arms)}
    out["frac_grad_worse_than_rank"] = {m: float(np.nanmean(GC[k] < RC[k])) for k, m in enumerate(arms)}
    out["raw_dv_grad_mean"] = {m: float(np.nanmean(DVG[k])) for k, m in enumerate(arms)}
    out["raw_dv_rank_mean"] = {m: float(np.nanmean(DVR[k])) for k, m in enumerate(arms)}
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"states {S}, goals {G}, arms {A}; dropped degenerate (s,g) cells: {n_dropped}")
    for name in ("grad_capture", "rank_capture", "gain_capture"):
        r = out[name]
        print(f"\n== {name}: Friedman p={r['friedman']['p']:.2g} W={r['friedman']['kendall_w']:.2f}; "
              f"CI-resolved {r['n_resolved_ci']}/15, Holm<.05 {r['n_holm_05']}/15; goal ICC {r['icc_goals']:.3f}; "
              f"sd G=1 {r['median_sd_G1']:.3f} -> G={G} {r['median_sd_Gall']:.3f} (shrink {r['sd_shrink']:.2f}x); "
              f"states for d=0.05: {r['req_states_d005']}")
        rs = r["rank_stability"]
        for m in sorted(arms, key=lambda m: -r["means"][m]):
            print(f"   {m:22s} {r['means'][m]:+.3f}  rank95 [{rs[m]['rank_lo']},{rs[m]['rank_hi']}] P1={rs[m]['p_first']:.2f}")
        for pp in sorted(r["pairs"], key=lambda q: q["p_holm"])[:6]:
            print(f"      {pp['a']:>22s} - {pp['b']:<22s} {pp['mean_diff']:+.3f} [{pp['ci_lo']:+.3f},{pp['ci_hi']:+.3f}] "
                  f"holm={pp['p_holm']:.3f} wins {pp['wins_a']}-{pp['wins_b']}")
    on = out["optimiser_noise"]
    print(f"\noptimiser noise (n={on['n']}): single-run sd {on['sd_single_run']:.3f} capture units; "
          f"mean |diff| {on['mean_abs_diff']:.3f}, max {on['max_abs_diff']:.3f}; per arm "
          + " ".join(f"{k}={v:.3f}" for k, v in on["per_arm"].items()))
    print(f"rank vs grad arm ordering: Kendall tau {out['rank_vs_grad_kendall']['tau']:+.2f} "
          f"(p={out['rank_vs_grad_kendall']['p']:.2f})")
    print("frac cells GD beats whole pool: " + " ".join(f"{k}={v:.2f}" for k, v in out["frac_grad_beats_pool"].items()))
    print("frac cells GD worse than rank pick: " + " ".join(f"{k}={v:.2f}" for k, v in out["frac_grad_worse_than_rank"].items()))
    print("bound-hit box/len: " + " ".join(f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in out["bound_hit"].items()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
