"""EXP-0037 stage 3: score everything from the banked outcomes (soft truth,
lyapunov per goal field) and test P1-P4 (DESIGN.md). Writes results/analysis.json.

Per (state s, goal g): the pool is the 128 re-simulated corpus pushes (same path
as the arms' pushes). capture(x) = (mean_pool - dv(x)) / (mean_pool - best_pool)
(lyapunov is a cost). grad = the restart with the best PREDICTED dv.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import best_of, higher_is_better_for, improvement
from Baselines.common.paired_stats import friedman, paired_comparison, rank_stability, variance_components
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.gt_bank import GroundTruthBank

ART = REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/artifacts/RUN-0001"
RES = REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/results/analysis.json"
HIB = higher_is_better_for("lyapunov")


def lyap_fields(occ, D):
    f = occ.reshape(len(occ), -1)
    return (f @ D.reshape(len(D), -1).T) / f.sum(1, keepdim=True).clamp_min(1e-6)


def cap(dv, pool):
    den = improvement(best_of(pool, higher_is_better=HIB), pool.mean(), higher_is_better=HIB)
    return float(improvement(dv, pool.mean(), higher_is_better=HIB) / den) if abs(den) > 1e-9 else np.nan


s1 = torch.load(ART / "stage1.pt", weights_only=False)
s2 = torch.load(ART / "stage2_index.pt", weights_only=False)
bank = GroundTruthBank(REPO / "datasets/DS-0004-ground-truth-bank/data")
D = s1["goal_fields"].float()
arms, goals, states = list(s1["arms"]), s1["goals"], sorted(s2["labels"])
A, G, S, R = len(arms), len(goals), len(states), s1["config"]["n_restart"]
RC = np.full((A, S, G), np.nan); GC = np.full((A, S, G), np.nan); RS = np.full((A, S, G, R), np.nan)
for si, s in enumerate(states):
    rec = bank.record("oracle_rollout_full", s2["state_keys"][s])
    L = s2["labels"][s]
    st = rec["state"]
    hit, fin = bank.lookup(st, L["actions"], "oracle_rollout_full"); assert hit.all()
    v0 = lyap_fields(occ_for_scoring(st[None, :, :3]), D)[0]
    dv = (lyap_fields(occ_for_scoring(fin[..., :3]), D) - v0[None]).numpy()          # (N, G)
    idx = {l: i for i, l in enumerate(map(tuple, L["labels"]))}
    pool_i = [i for i, l in enumerate(L["labels"]) if l[0] == "pool"]
    for gi in range(G):
        pool = dv[pool_i, gi]
        for ai, arm in enumerate(arms):
            RC[ai, si, gi] = cap(dv[idx[("rank", arm, gi, -1)], gi], pool)
            rs = [cap(dv[idx[("restart", arm, gi, r)], gi], pool) for r in range(R)]
            RS[ai, si, gi] = rs
            GC[ai, si, gi] = rs[int(s1["arms"][arm]["pred_restart"][s][gi].argmin())]
out = {"arms": arms, "goals": goals, "n_states": S, "n_restart": R}
for name, X3 in (("rank_capture", RC), ("grad_capture", GC), ("gain_capture", GC - RC)):
    X = np.nanmean(X3, 2)
    pairs = paired_comparison(X, arms)
    out[name] = dict(means={a: float(np.nanmean(X[k])) for k, a in enumerate(arms)},
                     friedman=friedman(X), variance_components=variance_components(X),
                     rank_stability=rank_stability(X, arms), pairs=pairs,
                     n_holm_05=sum(p["p_holm"] < .05 for p in pairs))
t = stats.kendalltau([out["rank_capture"]["means"][a] for a in arms], [out["grad_capture"]["means"][a] for a in arms])
out["P2_rank_vs_grad_kendall"] = dict(tau=float(t.statistic), p=float(t.pvalue))
spread = np.nanstd(RS, axis=3, ddof=1)                                                  # (A, S, G)
between = float(np.nanstd([out["grad_capture"]["means"][a] for a in arms], ddof=1))
out["P4_restart_spread"] = dict(median_sd=float(np.nanmedian(spread)), between_arm_sd=between,
                                ratio=float(np.nanmedian(spread) / between) if between > 0 else np.nan)
ens = arms.index("ensemble_nfd")
single = [a for a in arms if a != "ensemble_nfd"]
best_single = max(single, key=lambda a: out["grad_capture"]["means"][a])
d = np.nanmean(GC[ens] - GC[arms.index(best_single)], 1)
bs = [np.nanmean(d[np.random.default_rng(k).integers(0, S, S)]) for k in range(5000)]
out["P3_ensemble_vs_best_single"] = dict(best_single=best_single, diff=float(np.nanmean(d)),
                                          ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
out["frac_grad_beats_pool"] = {a: float(np.nanmean(GC[k] > 1)) for k, a in enumerate(arms)}
out["frac_grad_worse_than_rank"] = {a: float(np.nanmean(GC[k] < RC[k])) for k, a in enumerate(arms)}
out["bound_hit_box_len"] = {a: torch.stack(list(s1["arms"][a]["hits"].values())).float().mean((0, 1, 2)).tolist() for a in arms}
RES.parent.mkdir(parents=True, exist_ok=True); RES.write_text(json.dumps(out, indent=1))
for name in ("rank_capture", "grad_capture", "gain_capture"):
    r = out[name]
    print(f"\n== {name}: Friedman p={r['friedman']['p']:.2g}; Holm<.05 pairs {r['n_holm_05']}/{len(r['pairs'])}")
    for a in sorted(arms, key=lambda a: -r["means"][a]):
        rs_ = r["rank_stability"][a]
        print(f"   {a:40s} {r['means'][a]:+.3f}  rank95 [{rs_['rank_lo']},{rs_['rank_hi']}]")
print(f"\nP2 Kendall(rank_capture, grad_capture) over arms: {t.statistic:+.2f} (p {t.pvalue:.2f})")
print(f"P3 ensemble - best single ({best_single}) grad_capture: {out['P3_ensemble_vs_best_single']['diff']:+.4f} {out['P3_ensemble_vs_best_single']['ci']}")
print(f"P4 median restart sd {out['P4_restart_spread']['median_sd']:.3f} vs between-arm sd {between:.3f} (ratio {out['P4_restart_spread']['ratio']:.2f})")
print("frac cells GD beats whole pool:", {a: round(v, 2) for a, v in out["frac_grad_beats_pool"].items()})
print("frac cells GD worse than rank pick:", {a: round(v, 2) for a, v in out["frac_grad_worse_than_rank"].items()})
