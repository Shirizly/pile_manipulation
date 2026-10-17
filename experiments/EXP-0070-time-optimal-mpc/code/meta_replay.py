"""EXP-0070 RUN-0014 (meta-planner step 2): offline replay of deliberate-vs-act stopping rules on meta_traces.json.

Seconds-valued outcome of stopping at checkpoint k (incumbent = `mass` selection):
    N_true(k) = f(r0) - f(max(0, r0 - dM_true_k)) - t_act_k - tau_k
f = gap|td time-to-go map (etc_map_m1), r0 = the state's mass gap.  Policies:
  const@tau      stop at tau for every state
  oracle         per-state argmax_k N_true (unattainable bound)
  A schedule     r0-only: rho_k = E[dM_true_k / r0] (fraction of the remaining gap the tau_k incumbent closes),
                 N_hat(k) = f(r0) - f(r0 - r0 rho_k) - T_k - tau_k; stop at the first k with N_hat(k+1) <= N_hat(k)
                 (the value-of-computation rule: continue iff the next step buys more task time than it costs)
  B online       A, but the CURRENT gain is the state's own incumbent: g_k = kappa_k * (dM_pred_k + offset), kappa_k =
                 E[dM_true_k] / E[dM_pred_k + offset]; the expected NEXT increment is r0 (rho_{k+1} - rho_k)
Tables (rho, kappa, T) are cross-fitted: 2 folds by start state, fitted on the other fold, per goal family.
Also: the best constant budget chosen on the other fold (what a globally-tuned constant budget would really get).
-> results/meta_replay.json, results/meta_schedule.json (rule A's table fitted on ALL states, for the closed loop)
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
rows = json.loads((HERE / "results/meta_traces.json").read_text())
emap = json.loads((HERE / "results/etc_map_m1.json").read_text())["goals"]
CK = [c["tau"] for c in rows[0]["cps"]]
fam = lambda g: "quadrant" if g.startswith("quad") else ("two_squares" if g == "two_squares" else "letter")
rows = [r for r in rows if r["r0"] > 1e-6]                # already complete: nothing to decide
F = lambda g, r: emap[g]["gap|td"]["c"] * np.clip(r, 0, None) ** emap[g]["gap|td"]["p"]


def ntrue(r, k):
    c = r["cps"][k]["mass"]
    return F(r["goal"], r["r0"]) - F(r["goal"], max(0.0, r["r0"] - c["true_dM"])) - c["t"] - CK[k]


def tables(R):
    out = {}
    for f_ in set(fam(r["goal"]) for r in R) | {"ALL"}:
        S = [r for r in R if f_ == "ALL" or fam(r["goal"]) == f_]
        rho = [float(np.mean([min(1.5, r["cps"][k]["mass"]["true_dM"] / r["r0"]) for r in S])) for k in range(len(CK))]
        kap = [float(np.mean([r["cps"][k]["mass"]["true_dM"] for r in S]) /
                     max(np.mean([r["cps"][k]["mass"]["dM"] + r["off"] for r in S]), 1e-6)) for k in range(len(CK))]
        T = [float(np.mean([r["cps"][k]["mass"]["t"] for r in S])) for k in range(len(CK))]
        out[f_] = dict(rho=rho, kappa=kap, T=T, n=len(S))
    return out


def rule_A(r, tb):
    t = tb.get(fam(r["goal"]), tb["ALL"]); g = r["goal"]; r0 = r["r0"]
    nh = [F(g, r0) - F(g, r0 - r0 * min(1.0, t["rho"][k])) - t["T"][k] - CK[k] for k in range(len(CK))]
    k = 0
    while k + 1 < len(CK) and nh[k + 1] > nh[k]:
        k += 1
    return k


def rule_B(r, tb):
    t = tb.get(fam(r["goal"]), tb["ALL"]); g = r["goal"]; r0 = r["r0"]
    k = 0
    while k + 1 < len(CK):
        gk = min(r0, max(0.0, t["kappa"][k] * (r["cps"][k]["mass"]["dM"] + r["off"])))
        inc = max(0.0, r0 * (t["rho"][k + 1] - t["rho"][k]))
        if F(g, r0 - gk) - F(g, max(0.0, r0 - gk - inc)) > (CK[k + 1] - CK[k]) + (t["T"][k + 1] - t["T"][k]):
            k += 1
        else:
            break
    return k


def boot(d):
    d = np.asarray(d); bs = rng.choice(d, (4000, len(d))).mean(1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))


folds = np.array([r["start"] % 2 for r in rows])
choice = {"A": np.zeros(len(rows), int), "B": np.zeros(len(rows), int), "const_cv": np.zeros(len(rows), int)}
NT = np.array([[ntrue(r, k) for k in range(len(CK))] for r in rows])
for f_ in (0, 1):
    tr = [r for r, ff in zip(rows, folds) if ff != f_]; tb = tables(tr)
    kbest = int(np.argmax(NT[folds != f_].mean(0)))
    for i, r in enumerate(rows):
        if folds[i] == f_:
            choice["A"][i] = rule_A(r, tb); choice["B"][i] = rule_B(r, tb); choice["const_cv"][i] = kbest
res = dict(CK=CK, n_states=len(rows), curves={}, policies={})
for f_ in ("ALL", "quadrant", "two_squares", "letter"):
    S = [i for i, r in enumerate(rows) if f_ == "ALL" or fam(r["goal"]) == f_]
    res["curves"][f_] = dict(
        true_dM_mass=[float(np.mean([rows[i]["cps"][k]["mass"]["true_dM"] for i in S])) for k in range(len(CK))],
        pred_dM_mass_corr=[float(np.mean([rows[i]["cps"][k]["mass"]["dM"] + rows[i]["off"] for i in S])) for k in range(len(CK))],
        true_dV_lyap=[float(np.mean([rows[i]["cps"][k]["lyap"]["true_dV"] for i in S])) for k in range(len(CK))],
        pred_dV_lyap=[float(np.mean([rows[i]["cps"][k]["lyap"]["dv"] for i in S])) for k in range(len(CK))],
        true_dM_lyap=[float(np.mean([rows[i]["cps"][k]["lyap"]["true_dM"] for i in S])) for k in range(len(CK))],
        N_true_mean=[float(NT[S, k].mean()) for k in range(len(CK))],
        iters=[float(np.mean([rows[i]["cps"][k]["iters"] for i in S])) for k in range(len(CK))])
    pol = {f"const@{CK[k]}": NT[S, k] for k in range(len(CK))}
    pol["oracle"] = NT[S].max(1)
    for name in ("A", "B", "const_cv"):
        pol[name] = NT[S, choice[name][S]]
    best_k = int(np.argmax(NT[S].mean(0)))
    res["policies"][f_] = {name: dict(**boot(v), tau=(float(np.mean([CK[k] for k in choice[name][S]])) if name in choice else
                                                       (CK[best_k] if name == "oracle" else None)))
                           for name, v in pol.items()}
    res["policies"][f_]["A_minus_best_const_hindsight"] = boot(pol["A"] - NT[S, best_k])
    res["policies"][f_]["B_minus_best_const_hindsight"] = boot(pol["B"] - NT[S, best_k])
    res["policies"][f_]["A_minus_const_cv"] = boot(pol["A"] - pol["const_cv"])
    res["policies"][f_]["B_minus_const_cv"] = boot(pol["B"] - pol["const_cv"])
    res["policies"][f_]["best_const_hindsight_tau"] = CK[best_k]
(HERE / "results/meta_replay.json").write_text(json.dumps(res, indent=1))
# the closed-loop schedule: rule A's tables on ALL states, plus the resulting tau*(r0) per family on a grid of r0
tb = tables(rows)
grid = np.linspace(0.01, 0.9, 90)
sched = {f_: dict(r=grid.tolist(), tau=[CK[rule_A(dict(goal=g, r0=float(x)), tb)] for x in grid], goal_for_f=g)
         for f_, g in (("quadrant", "quadrant_0"), ("two_squares", "two_squares"), ("letter", "letter_O"))}
(HERE / "results/meta_schedule.json").write_text(json.dumps(dict(tables=tb, CK=CK, schedule_example=sched), indent=1))
for f_, c in res["curves"].items():
    print(f"== {f_}: iters {np.round(c['iters'], 1).tolist()}")
    print("   mass incumbent true dM  ", np.round(c["true_dM_mass"], 4).tolist())
    print("   mass incumbent pred(corr)", np.round(c["pred_dM_mass_corr"], 4).tolist())
    print("   lyap incumbent true dV  ", np.round(c["true_dV_lyap"], 4).tolist(), " pred", np.round(c["pred_dV_lyap"], 4).tolist())
    print("   N_true by tau           ", np.round(c["N_true_mean"], 3).tolist())
    for name, v in res["policies"][f_].items():
        if isinstance(v, dict):
            print(f"   {name:32s} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]" + (f"  tau {v['tau']:.3f}" if v.get('tau') is not None else ""))
    print("   best const (hindsight) tau", res["policies"][f_]["best_const_hindsight_tau"])
