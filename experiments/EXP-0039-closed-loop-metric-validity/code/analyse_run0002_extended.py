"""EXP-0039 RUN-0002 EXTENDED analysis (2026-10-04, zero-simulation, CPU). Provenance: grew out of the audit
recompute script (`recompute.py`, session scratchpad 2026-10-04) that re-derived RUN-0002's numbers from the
recorded episodes; everything here is computed from results/run0002/hm_{cem,rank}_*.json, results/run0002_analysis.json
(Holm-resolved pairs and the six PRE-REGISTERED offline metrics, unchanged) and the cached DS-0016 offline scores
(EXP-0059 results/test_v2.json + results/run0002/offline_ds0016_scores.json).

What it adds over analyse_run0002.py (all EXPLORATORY unless marked pre-registered):
  * every offline metric the caches hold for the 8 models (accuracy_1_blur1/2, _scatter, _clump, rollout_accuracy_k,
    occ_emd_swept, mass_in_goal_mae[_tough]) -- these were NOT in the pre-registered candidate list;
  * matched-letter rescore: slateN / within-pool Spearman / optimism / top-1 regret on the closed-loop letters that the
    cached pools cover (O T S X L = 5 of 8; Z C H were never scored offline, so an exact 8-letter rescore is impossible
    from caches);
  * per metric x arm: Spearman rho, Kendall tau, EXACT permutation p (n = 8 models, 8! = 40320 relabelings, one-sided
    in the predicted direction), Holm correction over the 6 pre-registered metrics within an arm and over all
    metrics x 2 arms, pair agreement with the within-training-seed pairs (narrow v2 seeds 0/1/2) excluded and included;
  * on-policy rows from the recorded pred_dv / true_dv of every executed push: per model corr, MAE, optimism (true - pred,
    cost sense, > 0 = improvement over-predicted), and their rank correlation with the closed-loop score;
  * executed un-legalisable pushes (legal_ok == False -> push executed unchanged) per model per arm, and the model order
    with those episodes dropped;
  * per-goal closed-loop scores and per-goal Kendall tau with the pooled order.
Usage: python -u code/analyse_run0002_extended.py -> results/run0002_extended.{json,md}
"""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_run0002 as A2  # noqa: E402  (episode_scores: in-goal mass / optimum over pushes 4..16)

HERE, R, REPO = A2.HERE, A2.R, A2.REPO
SEEDS = {"nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2"}
CL_LETTERS = ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z", "letter_C", "letter_H"]
PREREG = {"accuracy_1": True, "slateN": True, "slateN_tough": True, "spearman": True, "optimism": False, "top1_regret": False}
EXTRA = {"accuracy_1_blur1": True, "accuracy_1_blur2": True, "accuracy_1_scatter": True, "accuracy_1_clump": True,
         "rollout_accuracy_1": True, "rollout_accuracy_2": True, "rollout_accuracy_3": True, "rollout_accuracy_4": True,
         "occ_emd_swept": False, "mass_in_goal_mae": False, "mass_in_goal_mae_tough": False}
MATCHED = {"slateN_matched5": True, "spearman_matched5": True, "optimism_matched5": False, "top1_regret_matched5": False}
PERMS = np.array(list(itertools.permutations(range(8))))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def perm_p(x, y):
    """Spearman rho of x vs y and the exact one-sided permutation p (fraction of the n! relabelings of y with rho >= observed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert len(x) == len(PERMS[0]) == len(y)
    rx = rankdata(x); ry = rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    obs = float(np.mean(rx * ry))
    allr = (ry[PERMS] * rx).mean(1)
    return obs, float(np.mean(allr >= obs - 1e-12))


def holm(ps):
    """Holm-adjusted p-values (dict name -> p)."""
    names = sorted(ps, key=lambda k: ps[k]); n = len(names); adj = {}; run = 0.0
    for r, k in enumerate(names):
        run = max(run, min(1.0, ps[k] * (n - r))); adj[k] = run
    return adj


def load_src():
    src = {}
    for f in (REPO / "experiments/EXP-0059-retrieval-transition-model/results/test_v2.json", R / "offline_ds0016_scores.json"):
        src.update({k: v for k, v in json.loads(f.read_text()).items() if isinstance(v, dict) and "metrics" in v})
    return src


def matched_letter_metrics(src, models):
    """slateN capture, within-pool Spearman, optimism and top-1 regret on the closed-loop letters the cached pools hold."""
    out = {}
    for m in models:
        raw = src[m]["raw"]["slateN_raw"]; goals = [g for g in CL_LETTERS if g in raw]
        caps, sp, op, rg, per_goal = [], [], [], [], {}
        for g in goals:
            cg = []
            for p in raw[g]:
                vt, vp = np.asarray(p["vt"]), np.asarray(p["vp"]); den = vt.mean() - vt.min()
                if den > 1e-9:
                    cg.append((vt.mean() - vt[vp.argmin()]) / den)
                if np.ptp(vp) > 0 and np.ptp(vt) > 0:
                    sp.append(spearmanr(vp, vt)[0])
                i = int(np.argmin(vp)); op.append(vt[i] - vp[i]); rg.append(vt[i] - vt.min())
            caps += cg; per_goal[g] = float(np.mean(cg))
            # consistency check against the cached per-goal slateN
            cached = src[m]["metrics"].get("slateN_per_goal", {}).get(g)
            if cached is not None:
                assert abs(cached - per_goal[g]) < 1e-6, (m, g, cached, per_goal[g])
        out[m] = dict(goals=goals, slateN_matched5=float(np.mean(caps)), spearman_matched5=float(np.mean(sp)),
                      optimism_matched5=float(np.mean(op)), top1_regret_matched5=float(np.mean(rg)), slateN_per_letter=per_goal)
    return out


def load_arm(arm):
    files = sorted(glob.glob(str(R / f"hm_{arm}_*.json")))
    eps, onp, illegal = {}, {}, {}
    for f in files:
        m = Path(f).stem.split(f"hm_{arm}_", 1)[1]
        eps[m] = A2.episode_scores(f)
        pd_, td, bad, n = [], [], {}, 0
        for e in json.loads(Path(f).read_text())["episodes"]:
            if not e.get("complete"):
                continue
            pd_ += e["pred_dv"]; td += e["true_dv"]
            nb = sum(not x for x in e["legal_ok"]); n += len(e["legal_ok"])
            bad[(e["goal"], e["start"])] = nb
        pd_, td = np.array(pd_, float), np.array(td, float)
        onp[m] = dict(corr=float(np.corrcoef(pd_, td)[0, 1]), mae=float(np.mean(np.abs(pd_ - td))),
                      optimism=float(np.mean(td - pd_)), abs_optimism=float(abs(np.mean(td - pd_))),
                      pred_dv_mean=float(pd_.mean()), true_dv_mean=float(td.mean()), n_pushes=int(len(td)))
        illegal[m] = dict(n_pushes=n, n_ok_false=int(sum(bad.values())), rate=float(sum(bad.values()) / n),
                          episodes_with_any=int(sum(v > 0 for v in bad.values())), bad_by_episode=bad)
    return eps, onp, illegal


def analyse_arm(arm, ana, src, matched):
    eps, onp, illegal = load_arm(arm)
    models = sorted(eps); keys = sorted(set.intersection(*[set(v) for v in eps.values()]))
    X = np.array([[eps[m][k]["score"] for k in keys] for m in models]); cl = X.mean(1)
    rep = np.array([ana[arm]["per_model"][m]["score"] for m in models])
    res = dict(models=models, n_episodes=len(keys), closed_loop={m: float(cl[i]) for i, m in enumerate(models)},
               max_abs_diff_vs_reported=float(np.max(np.abs(cl - rep))))
    pairs = ana[arm]["pairs"]; resolved = [q for q in pairs if q["holm_resolved"]]
    within = [q for q in resolved if q["a"] in SEEDS and q["b"] in SEEDS]
    res["n_resolved"] = len(resolved); res["n_resolved_within_seed"] = len(within)
    res["within_seed_resolved"] = [f"{q['a']} {'>' if q['diff'] > 0 else '<'} {q['b']} ({q['diff']:+.3f})" for q in within]
    seed_scores = [res["closed_loop"][s] for s in SEEDS]; res["seed_spread"] = float(max(seed_scores) - min(seed_scores))
    # offline metric values
    off = {}
    for m in models:
        off[m] = dict(ana[arm]["offline"][m])
        off[m].update({k: src[m]["metrics"][k] for k in EXTRA})
        off[m].update({k: matched[m][k] for k in MATCHED})
    res["offline"] = off
    # metric -> closed-loop correlations
    groups = [("pre-registered", PREREG), ("exploratory-cached", EXTRA), ("exploratory-matched-letters", MATCHED)]
    met = {}
    for grp, dct in groups:
        for k, hib in dct.items():
            mv = np.array([off[m][k] for m in models], float) * (1 if hib else -1)
            rho, p = perm_p(mv, cl); tau = kendalltau(mv, cl)[0]
            def agree(qs):
                a = [((off[q["a"]][k] - off[q["b"]][k]) * (1 if hib else -1) * q["diff"]) > 0 for q in qs]
                return (float(np.mean(a)) if a else None, len(a))
            inc = agree(resolved); exc = agree([q for q in resolved if not (q["a"] in SEEDS and q["b"] in SEEDS)])
            met[k] = dict(group=grp, higher_is_better=hib, rho=rho, tau=float(tau), perm_p_one_sided=p,
                          pair_agree_incl_seed=inc[0], n_pairs_incl=inc[1], pair_agree_excl_seed=exc[0], n_pairs_excl=exc[1])
    # on-policy rows (per model values + rank correlation with closed loop)
    for k, hib in (("corr", True), ("mae", False), ("optimism", False), ("abs_optimism", False)):
        v = np.array([onp[m][k] for m in models]) * (1 if hib else -1)
        rho, p = perm_p(v, cl); tau = kendalltau(v, cl)[0]
        def agree(qs):
            a = [((onp[q["a"]][k] - onp[q["b"]][k]) * (1 if hib else -1) * q["diff"]) > 0 for q in qs]
            return (float(np.mean(a)) if a else None, len(a))
        inc = agree(resolved); exc = agree([q for q in resolved if not (q["a"] in SEEDS and q["b"] in SEEDS)])
        met["onpolicy_" + k] = dict(group="exploratory-on-policy", higher_is_better=hib, rho=rho, tau=float(tau), perm_p_one_sided=p,
                                     pair_agree_incl_seed=inc[0], n_pairs_incl=inc[1], pair_agree_excl_seed=exc[0], n_pairs_excl=exc[1])
    # multiplicity: Holm over the 6 pre-registered metrics within the arm; over everything in the arm
    hp = holm({k: met[k]["perm_p_one_sided"] for k in PREREG})
    ha = holm({k: met[k]["perm_p_one_sided"] for k in met})
    for k in met:
        met[k]["holm_p_prereg6"] = hp.get(k); met[k]["holm_p_all_in_arm"] = ha[k]
    res["metrics"] = met; res["onpolicy"] = onp
    # executed un-legalisable pushes and order robustness
    res["illegal_executed"] = {m: {k: v for k, v in illegal[m].items() if k != "bad_by_episode"} for m in models}
    tot_n = sum(illegal[m]["n_pushes"] for m in models); tot_b = sum(illegal[m]["n_ok_false"] for m in models)
    res["illegal_executed_total"] = dict(n_pushes=tot_n, n_ok_false=tot_b, rate=float(tot_b / tot_n))
    bad_keys = {k for m in models for k, v in illegal[m]["bad_by_episode"].items() if v > 0}
    keep = [i for i, k in enumerate(keys) if k not in bad_keys]
    cl_clean = X[:, keep].mean(1) if keep else np.full(len(models), np.nan)
    # (a) paired drop: an episode is dropped for every model if ANY model executed an ok=False push in it (keeps pairing, loses most episodes);
    # (b) per-model drop: each model keeps its own clean episodes (unpaired means, keeps most data)
    pm_clean = np.array([np.mean([X[i, j] for j, k in enumerate(keys) if illegal[m]["bad_by_episode"][k] == 0]) for i, m in enumerate(models)])
    res["order_dropping_illegal_episodes"] = dict(n_episodes_kept=len(keep), n_dropped=len(keys) - len(keep),
                                                   closed_loop_clean={m: float(cl_clean[i]) for i, m in enumerate(models)},
                                                   kendall_tau_vs_full=float(kendalltau(cl, cl_clean)[0]) if keep else None,
                                                   per_model_drop_clean={m: float(pm_clean[i]) for i, m in enumerate(models)},
                                                   per_model_drop_n_kept={m: int(sum(v == 0 for v in illegal[m]["bad_by_episode"].values())) for m in models},
                                                   per_model_drop_kendall_tau_vs_full=float(kendalltau(cl, pm_clean)[0]))
    # per-goal
    pg = {}
    for g in CL_LETTERS:
        gi = [i for i, k in enumerate(keys) if k[0] == g]
        if not gi:
            continue
        s = X[:, gi].mean(1)
        pg[g] = dict(n_episodes=len(gi), scores={m: float(s[i]) for i, m in enumerate(models)}, kendall_tau_vs_pooled=float(kendalltau(s, cl)[0]),
                     mean_over_models=float(s.mean()))
    res["per_goal"] = pg
    # per-goal slateN (cached per-goal, the 5 covered letters) vs per-goal closed-loop order
    pgs = {}
    for g in CL_LETTERS:
        if g in pg and all(g in src[m]["metrics"].get("slateN_per_goal", {}) for m in models):
            mv = np.array([src[m]["metrics"]["slateN_per_goal"][g] for m in models]); s = np.array([pg[g]["scores"][m] for m in models])
            rho, p = perm_p(mv, s); pgs[g] = dict(rho_slateN_goal_vs_cl_goal=rho, perm_p=p)
    res["per_goal_slateN_vs_closed_loop"] = pgs
    return res


def fmt_md(out):
    L = ["# EXP-0039 RUN-0002 extended analysis (generated by code/analyse_run0002_extended.py, 2026-10-04)\n",
         "Exploratory except where marked pre-registered. n = 8 models; permutation p = exact one-sided (8! relabelings) in the",
         "predicted direction; Holm columns correct over the 6 pre-registered metrics within an arm / over every row in the arm.",
         "Pair agreement = fraction of Holm-resolved closed-loop pairs ordered correctly, with the within-seed pairs (narrow v2",
         "seeds 0/1/2) excluded (excl) and included (incl). `*_matched5` = rescored on the 5 closed-loop letters the DS-0016",
         "caches hold (O T S X L); Z C H have no cached offline pools, so an exact 8-letter rescore is not possible without GPU.\n"]
    for arm, r in out["arms"].items():
        L.append(f"\n## arm {arm}: {r['n_episodes']} paired episodes, {r['n_resolved']} Holm-resolved pairs "
                 f"({r['n_resolved_within_seed']} within-seed: {'; '.join(r['within_seed_resolved']) or 'none'}); "
                 f"seed spread {r['seed_spread']:.3f}; recomputed vs reported max |diff| {r['max_abs_diff_vs_reported']:.1e}\n")
        L.append("| metric | group | rho | tau | perm p | Holm p (prereg 6) | Holm p (all) | agree excl seed | agree incl seed |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in sorted(r["metrics"].items(), key=lambda kv: (kv[1]["group"], -kv[1]["rho"])):
            hp = f"{v['holm_p_prereg6']:.3f}" if v["holm_p_prereg6"] is not None else "-"
            L.append(f"| {k}{'' if v['higher_is_better'] else ' (flipped)'} | {v['group']} | {v['rho']:+.2f} | {v['tau']:+.2f} | {v['perm_p_one_sided']:.3f} | {hp} | {v['holm_p_all_in_arm']:.3f} | "
                     f"{v['pair_agree_excl_seed']:.2f} of {v['n_pairs_excl']} | {v['pair_agree_incl_seed']:.2f} of {v['n_pairs_incl']} |")
        L.append("\n| model | closed loop | on-policy corr | on-policy MAE | on-policy optimism (true-pred) | ok=False pushes | rate | clean score, paired drop | clean score, per-model drop (n kept) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m in sorted(r["models"], key=lambda m: -r["closed_loop"][m]):
            o, il = r["onpolicy"][m], r["illegal_executed"][m]
            L.append(f"| {m} | {r['closed_loop'][m]:.3f} | {o['corr']:+.3f} | {o['mae']:.4f} | {o['optimism']:+.4f} | {il['n_ok_false']}/{il['n_pushes']} | {il['rate']:.3f} | {r['order_dropping_illegal_episodes']['closed_loop_clean'][m]:.3f} | {r['order_dropping_illegal_episodes']['per_model_drop_clean'][m]:.3f} ({r['order_dropping_illegal_episodes']['per_model_drop_n_kept'][m]}) |")
        od = r["order_dropping_illegal_episodes"]; it = r["illegal_executed_total"]
        L.append(f"\nexecuted un-legalisable pushes: {it['n_ok_false']}/{it['n_pushes']} = {100 * it['rate']:.1f} %; dropping the {od['n_dropped']} episodes with any such push "
                 f"(keeping {od['n_episodes_kept']}): Kendall tau vs full order {od['kendall_tau_vs_full']:+.2f}; per-model drop (unpaired): tau {od['per_model_drop_kendall_tau_vs_full']:+.2f}")
        L.append("\n| goal | n eps | mean over models | tau vs pooled order | " + " | ".join(r["models"]) + " |")
        L.append("|---" * (4 + len(r["models"])) + "|")
        for g, v in r["per_goal"].items():
            L.append(f"| {g} | {v['n_episodes']} | {v['mean_over_models']:.3f} | {v['kendall_tau_vs_pooled']:+.2f} | " + " | ".join(f"{v['scores'][m]:.3f}" for m in r["models"]) + " |")
        if r["per_goal_slateN_vs_closed_loop"]:
            L.append("\nper-goal cached slateN vs per-goal closed-loop score (rho, perm p): " + "; ".join(f"{g} {v['rho_slateN_goal_vs_cl_goal']:+.2f} (p {v['perm_p']:.2f})" for g, v in r["per_goal_slateN_vs_closed_loop"].items()))
    L.append("\n## offline metric values (DS-0016 caches; matched5 = letters O T S X L)\n")
    arm0 = next(iter(out["arms"].values())); ks = list(PREREG) + list(EXTRA) + list(MATCHED)
    L.append("| model | " + " | ".join(ks) + " |"); L.append("|---" * (1 + len(ks)) + "|")
    for m in arm0["models"]:
        L.append(f"| {m} | " + " | ".join(f"{arm0['offline'][m][k]:.4f}" for k in ks) + " |")
    return "\n".join(L) + "\n"


def main():
    ana = json.loads((HERE / "results/run0002_analysis.json").read_text()); src = load_src()
    models = sorted(ana["cem"]["models"]); matched = matched_letter_metrics(src, models)
    out = dict(note=__doc__.strip().splitlines()[0], matched_letters=matched[models[0]]["goals"],
               slateN_tough_goal_overlap_with_closed_loop=sorted(set(CL_LETTERS) & set(src[models[0]]["raw"]["slateN_tough_raw"])), arms={})
    for arm in ("cem", "rank"):
        out["arms"][arm] = analyse_arm(arm, ana, src, matched)
    out = jsonable(out)
    tmp = HERE / "results/run0002_extended.json.tmp"; tmp.write_text(json.dumps(out, indent=1)); tmp.replace(HERE / "results/run0002_extended.json")
    md = fmt_md(out); (HERE / "results/run0002_extended.md").write_text(md); print(md)


if __name__ == "__main__":
    main()
