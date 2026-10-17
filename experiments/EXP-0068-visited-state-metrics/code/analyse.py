"""EXP-0068 analysis (pre-registered in DESIGN.md; CPU, zero simulation).

Inputs: results/scores_ds0023.json (visited states, DS-0023) and results/scores_ds0016.json (spawn pools, DS-0016) from
score_pools.py; closed-loop references: EXP-0039 results/run0002_analysis.json (CEM / rank arms: per-model scores,
Holm-resolved pairs) and EXP-0066 results/<cell>_ps0.json (8 planner-seed-0 cells, episode scores via EXP-0066
analyse.load_cell, Holm over the 28 pairs with analyse_run0002's bootstrap rule).
Per (arm, aggregation, metric): Spearman rho with the closed-loop order, exact one-sided permutation p (n! relabelings,
analyse_run0002_extended.perm_p), Kendall tau, pair agreement over Holm-resolved pairs with the within-seed pairs
excluded / included, and a state-bootstrap 95 % CI on rho (resample pools, 2000 draws).
H1: rho_visited(primary) > rho_spawn(primary) for >= 6 of 8 metrics in both RUN-0002 arms. H2: visited accuracy_1 under
CEM: rho >= 0.5 and pair agreement (seed pairs excluded) >= 0.70. Plus the per-model generalisation gap.
Usage: CUDA_VISIBLE_DEVICES= python -u code/analyse.py -> results/analysis.{json,md}
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/EXP-0068-visited-state-metrics"; RES = HERE / "results"
E39 = REPO / "experiments/EXP-0039-closed-loop-metric-validity"; E66 = REPO / "experiments/EXP-0066-closed-loop-power-study"
sys.path.insert(0, str(E66 / "code"))
LETTERS = ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z", "letter_C", "letter_H"]
OTSX = ["letter_O", "letter_T", "letter_S", "letter_X"]
ZOO = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2", "nfd_3ch_narrow_l20_v2_soft_s2",
       "nfd_3ch_narrow_l20_v2_epoch10", "linear_narrow_l20_v2_res64", "linear_switched_soft", "nfd_3ch_randlen"]
E66_CELLS = {"nfd_s0_ps0": "nfd_3ch_narrow_l20_v2", "nfd_s1_ps0": "nfd_3ch_narrow_l20_v2_seed1", "nfd_s2_ps0": "nfd_3ch_narrow_l20_v2_seed2",
             "lin_orig_ps0": "linear_narrow_l20_v2_res64", "lin_b1_ps0": "linear_narrow_l20_v2_res64_boot1",
             "lin_b2_ps0": "linear_narrow_l20_v2_res64_boot2", "broad_s0_ps0": "nfd_3ch_randlen", "broad_s1_ps0": "nfd_3ch_randlen_seed1"}
SEED_GROUPS = [{"nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2"},
               {"linear_narrow_l20_v2_res64", "linear_narrow_l20_v2_res64_boot1", "linear_narrow_l20_v2_res64_boot2"},
               {"nfd_3ch_randlen", "nfd_3ch_randlen_seed1"}]
METRICS = {"accuracy_1": True, "accuracy_1_blur2": True, "occ_emd_swept": False, "mass_in_goal_mae": False,
           "slateN": True, "top1_regret": False, "optimism": False, "spearman": True}
rng = np.random.default_rng(0)
B_BOOT = 2000


def within_seed(a, b):
    return any(a in g and b in g for g in SEED_GROUPS)


_PERMS = {}


def perm_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))))
    rx = rankdata(x); ry = rankdata(y); rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    obs = float(np.mean(rx * ry)); allr = (ry[_PERMS[n]] * rx).mean(1)
    return obs, float(np.mean(allr >= obs - 1e-12))


def holm_pairs(X, models):
    """X (M, E) episode scores -> list of pairs with bootstrap p and Holm resolution (analyse_run0002's rule)."""
    pv = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = X[i] - X[j]; bm = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
            p = float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean())))
            pv.append((p, i, j, float(d.mean())))
    order = sorted(range(len(pv)), key=lambda t: pv[t][0]); n = len(pv); still = True; out = []
    for r, t in enumerate(order):
        p, i, j, dm = pv[t]; still = still and p * (n - r) < 0.05
        out.append(dict(a=models[i], b=models[j], diff=dm, p=p, holm_resolved=bool(still)))
    return out


def load_arms():
    arms = {}
    a39 = json.loads((E39 / "results/run0002_analysis.json").read_text())
    for arm in ("cem", "rank"):
        arms[arm] = dict(models=sorted(a39[arm]["models"]), score={m: a39[arm]["per_model"][m]["score"] for m in a39[arm]["models"]},
                         pairs=a39[arm]["pairs"], n_episodes=a39[arm]["n_episodes"], source="EXP-0039 RUN-0002")
    try:
        import analyse as A66
        cells = {t: A66.load_cell(t) for t in E66_CELLS}
        keys = sorted(set.intersection(*[set(v) for v in cells.values()]))
        models = [E66_CELLS[t] for t in E66_CELLS]
        X = np.array([[cells[t][k]["score"] for k in keys] for t in E66_CELLS])
        arms["e66"] = dict(models=sorted(models), score={E66_CELLS[t]: float(X[i].mean()) for i, t in enumerate(E66_CELLS)},
                           pairs=holm_pairs(X, models), n_episodes=len(keys), source="EXP-0066 RUN-0001 (planner seed 0 cells)")
    except Exception as e:  # noqa: BLE001
        print("EXP-0066 arm unavailable:", e)
    return arms


class Corpus:
    """Per-pool, per-model contributions so any subset / bootstrap aggregates cheaply."""

    def __init__(self, path):
        s = json.loads(Path(path).read_text()); c = s["_corpus"]
        self.c = c; self.models = [m for m in s if m != "_corpus"]; self.s = s
        self.pool_idx = np.array(c["pool_idx"]); self.pools = sorted(set(self.pool_idx.tolist()))
        self.source = np.array(c["source"]); self.legal = np.array(c["legal_ok"], bool); self.valid = np.array(c["valid"], bool)
        self.den = np.array(c["den"]); self.den_b2 = np.array(c["den_b2"])
        self.vt = {g: np.array(c["vt"][g]) for g in LETTERS}
        self.meta = c.get("meta"); self.kind = c.get("start_kind")
        self.pool_goal = {p: (self.meta[i]["goal"] if self.meta else None) for i, p in enumerate(self.pools)}
        self.pool_k = {p: (self.meta[i]["k"] if self.meta else None) for i, p in enumerate(self.pools)}
        self.pool_model = {p: (self.meta[i]["model"] if self.meta else None) for i, p in enumerate(self.pools)}
        # DS-0016 stores start_kind per POOL (32 entries); DS-0023 per row
        self.pool_kind = {p: ((self.kind[i] if len(self.kind) == len(self.pools) else self.kind[int(np.nonzero(self.pool_idx == p)[0][0])])
                              if self.kind else None) for i, p in enumerate(self.pools)}

    def rows(self, p, mask):
        return np.nonzero((self.pool_idx == p) & mask)[0]

    def contributions(self, model, mask, goals_of_pool):
        """-> per pool dict of sums/values: num, den, num_b2, den_b2, emd_sum, n, and per (pool, goal) lists."""
        sm = self.s[model]; num = np.array(sm["num"]); num_b2 = np.array(sm["num_b2"]); emd = np.array(sm["emd"])
        out = {}
        for p in self.pools:
            ix = self.rows(p, mask); gl = goals_of_pool(p)
            if len(ix) == 0 or not gl:
                continue
            d = dict(num=num[ix].sum(), den=self.den[ix].sum(), num_b2=num_b2[ix].sum(), den_b2=self.den_b2[ix].sum(),
                     emd=emd[ix].sum(), n=len(ix), cap=[], sp=[], opt=[], reg=[], mass=[])
            for g in gl:
                vt = self.vt[g][ix]; vp = np.array(sm["vp"][g])[ix]; den = vt.mean() - vt.min()
                i = int(np.argmin(vp))
                if den > 1e-9:
                    d["cap"].append((vt.mean() - vt[i]) / den)
                if np.ptp(vp) > 0 and np.ptp(vt) > 0:
                    d["sp"].append(spearmanr(vp, vt)[0])
                d["opt"].append(vt[i] - vp[i]); d["reg"].append(vt[i] - vt.min())
                d["mass"].append(np.array(sm["mass_abs_err"][g])[ix].mean())
            out[p] = d
        return out


def aggregate(contribs, pools):
    """contribs: {pool: d}; pools: list (with repeats for a bootstrap draw) -> metric dict."""
    ds = [contribs[p] for p in pools if p in contribs]
    f = lambda k: float(np.mean([x for d in ds for x in d[k]])) if any(d[k] for d in ds) else float("nan")
    num = sum(d["num"] for d in ds); den = sum(d["den"] for d in ds); nb = sum(d["num_b2"] for d in ds); db = sum(d["den_b2"] for d in ds)
    return dict(accuracy_1=float(1 - np.sqrt(num / max(den, 1e-12))), accuracy_1_blur2=float(1 - np.sqrt(nb / max(db, 1e-12))),
                occ_emd_swept=float(sum(d["emd"] for d in ds) / max(sum(d["n"] for d in ds), 1)),
                mass_in_goal_mae=f("mass"), slateN=f("cap"), top1_regret=f("reg"), optimism=f("opt"), spearman=f("sp"),
                n_rows=int(sum(d["n"] for d in ds)), n_pools=len(ds))


def metric_vs_arm(vals, arm, metric, hib, boots=None):
    """vals: {model: value} -> rho / tau / perm p / pair agreement vs the arm's closed loop."""
    ms = [m for m in arm["models"] if m in vals and vals[m] is not None and not np.isnan(vals[m])]
    if len(ms) < 4:
        return None
    mv = np.array([vals[m] for m in ms]) * (1 if hib else -1); cl = np.array([arm["score"][m] for m in ms])
    rho, p = perm_p(mv, cl); tau = float(kendalltau(mv, cl)[0])
    res = [q for q in arm["pairs"] if q["holm_resolved"] and q["a"] in vals and q["b"] in vals]
    agree = lambda qs: (float(np.mean([((vals[q["a"]] - vals[q["b"]]) * (1 if hib else -1) * q["diff"]) > 0 for q in qs])) if qs else None, len(qs))
    inc = agree(res); exc = agree([q for q in res if not within_seed(q["a"], q["b"])])
    out = dict(n_models=len(ms), rho=rho, tau=tau, perm_p=p, agree_excl_seed=exc[0], n_pairs_excl=exc[1], agree_incl_seed=inc[0], n_pairs_incl=inc[1])
    if boots is not None:
        rs = []
        for bv in boots:
            x = np.array([bv[m] for m in ms]) * (1 if hib else -1)
            rs.append(spearmanr(x, cl)[0] if np.ptp(x) > 0 else np.nan)
        out["rho_ci"] = [float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))]
    return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RES; dst = Path(sys.argv[2]) if len(sys.argv) > 2 else RES   # dry runs
    arms = load_arms()
    vis = Corpus(src / "scores_ds0023.json"); spw = Corpus(src / "scores_ds0016.json")
    models = [m for m in ZOO + ["linear_narrow_l20_v2_res64_boot1", "linear_narrow_l20_v2_res64_boot2", "nfd_3ch_randlen_seed1"]
              if m in vis.models and m in spw.models]
    sampler = (vis.source == 0) & vis.valid
    plus = ((vis.source == 0) | ((vis.source == 1) & vis.legal)) & vis.valid
    aggs = {  # name: (corpus, row mask, goals_of_pool, pool filter)
        "visited_primary": (vis, sampler, lambda p: [vis.pool_goal[p]], lambda p: True),
        "visited_all8": (vis, sampler, lambda p: LETTERS, lambda p: True),
        "visited_plus_planner": (vis, plus, lambda p: [vis.pool_goal[p]], lambda p: True),
        "visited_OTSX_matched": (vis, sampler, lambda p: [vis.pool_goal[p]], lambda p: vis.pool_goal[p] in OTSX),
        "visited_k4": (vis, sampler, lambda p: [vis.pool_goal[p]], lambda p: vis.pool_k[p] == 4),
        "visited_k8": (vis, sampler, lambda p: [vis.pool_goal[p]], lambda p: vis.pool_k[p] == 8),
        "visited_k12": (vis, sampler, lambda p: [vis.pool_goal[p]], lambda p: vis.pool_k[p] == 12),
        "spawn_primary": (spw, spw.valid, lambda p: LETTERS, lambda p: True),
        "spawn_scatter": (spw, spw.valid, lambda p: LETTERS, lambda p: spw.pool_kind[p] == "scatter"),
        "spawn_OTSX": (spw, spw.valid, lambda p: OTSX, lambda p: True),
    }
    values, boots, contribs = {}, {}, {}
    for name, (corp, mask, gof, pf) in aggs.items():
        pools = [p for p in corp.pools if pf(p)]
        contribs[name] = {m: corp.contributions(m, mask, gof) for m in models}
        values[name] = {m: aggregate(contribs[name][m], pools) for m in models}
        draws = [list(rng.choice(pools, len(pools), replace=True)) for _ in range(B_BOOT)]
        boots[name] = {k: [{m: aggregate(contribs[name][m], d)[k] for m in models} for d in draws] for k in METRICS}
        print("aggregated", name, "pools", len(pools), "rows", values[name][models[0]]["n_rows"], flush=True)
    # metric-vs-arm table
    table = {}
    for arm_name, arm in arms.items():
        table[arm_name] = {}
        for name in aggs:
            table[arm_name][name] = {}
            for k, hib in METRICS.items():
                table[arm_name][name][k] = metric_vs_arm({m: values[name][m][k] for m in models}, arm, k, hib, boots[name][k])
    # H1 / H2
    def h1(arm_name, v, s):
        t = table[arm_name]; wins = [k for k in METRICS if t[v][k] and t[s][k] and t[v][k]["rho"] > t[s][k]["rho"]]
        aw = [k for k in METRICS if t[v][k] and t[s][k] and t[v][k]["agree_excl_seed"] is not None and t[s][k]["agree_excl_seed"] is not None
              and t[v][k]["agree_excl_seed"] > t[s][k]["agree_excl_seed"]]
        return dict(visited=v, spawn=s, n_metrics_rho_visited_higher=len(wins), metrics_rho_visited_higher=wins,
                    n_metrics_agree_visited_higher=len(aw), metrics_agree_visited_higher=aw)
    H1 = {a: h1(a, "visited_primary", "spawn_primary") for a in ("cem", "rank") if a in table}
    if "e66" in table:
        H1["e66"] = h1("e66", "visited_OTSX_matched", "spawn_OTSX")
    n_cem, n_rank = H1.get("cem", {}).get("n_metrics_rho_visited_higher", 0), H1.get("rank", {}).get("n_metrics_rho_visited_higher", 0)
    H1["verdict"] = "supported" if n_cem >= 6 and n_rank >= 6 else "refuted" if (n_cem <= 4 or n_rank <= 4) else "inconclusive"
    a1 = table["cem"]["visited_primary"]["accuracy_1"]; a1s = table["cem"]["spawn_primary"]["accuracy_1"]
    H2 = dict(rho_visited=a1["rho"], agree_excl_seed_visited=a1["agree_excl_seed"], rho_spawn=a1s["rho"], agree_excl_seed_spawn=a1s["agree_excl_seed"],
              verdict="supported" if (a1["rho"] >= 0.5 and (a1["agree_excl_seed"] or 0) >= 0.70)
              else "refuted" if (a1["rho"] < 0.3 or (a1["agree_excl_seed"] or 0) < 0.60) else "inconclusive")
    # generalisation gap
    gap = {m: values["spawn_primary"][m]["accuracy_1"] - values["visited_primary"][m]["accuracy_1"] for m in models}
    gap_vs = {a: metric_vs_arm(gap, arm, "gap", False) for a, arm in arms.items()}
    # DS-0023 descriptives: planner candidates vs sampler candidates in TRUTH (goal-matched dv), legality, null rows
    def _max_drift():
        mp = REPO / "datasets/DS-0023-visited-state-pools/data/manifest.json"
        if not mp.exists():
            return None
        v = [x.get("settle_drift_m", 0.0) for x in json.loads(mp.read_text())["states"].values()]
        return float(max(v) * 1000) if v else None
    desc = {}
    if vis.meta:
        md = np.array(vis.c["max_disp_mm"])
        pl_dv, sm_dv, pl_best = [], [], []
        for i, p in enumerate(vis.pools):
            g = vis.pool_goal[p]; ixs = vis.rows(p, sampler); ixp = vis.rows(p, (vis.source == 1) & vis.legal & vis.valid)
            if len(ixp):
                pl_dv.append(vis.vt[g][ixp].mean()); sm_dv.append(vis.vt[g][ixs].mean()); pl_best.append(vis.vt[g][ixp].min() <= vis.vt[g][ixs].min())
        desc = dict(n_states=len(vis.pools), n_rows=len(vis.pool_idx), n_sampler_rows=int(sampler.sum()),
                    n_planner_rows=int((vis.source == 1).sum()), n_planner_legal=int(((vis.source == 1) & vis.legal).sum()),
                    n_planner_valid_length=int(((vis.source == 1) & vis.valid).sum()),
                    null_rows_sampler=int((md[sampler] < 1.0).sum()), null_rows_planner=int((md[vis.source == 1] < 1.0).sum()),
                    n_layer1_states=int(sum(mm["n_layer1"] > 0 for mm in vis.meta)), n_illegal_states=int(sum(not mm["legal_state"] for mm in vis.meta)),
                    per_model_states={m: int(sum(mm["model"] == m for mm in vis.meta)) for m in ZOO},
                    planner_vs_sampler_true_dv=dict(mean_true_dv_planner=float(np.mean(pl_dv)) if pl_dv else None,
                                                    mean_true_dv_sampler=float(np.mean(sm_dv)) if sm_dv else None,
                                                    frac_states_planner_holds_true_best=float(np.mean(pl_best)) if pl_best else None,
                                                    n_states=len(pl_dv)),
                    settle_drift_mm_max=_max_drift())
    out = dict(models=models, arms={a: dict(models=v["models"], score=v["score"], n_episodes=v["n_episodes"], source=v["source"],
                                            n_resolved=sum(q["holm_resolved"] for q in v["pairs"])) for a, v in arms.items()},
               values=values, table=table, H1=H1, H2=H2, generalisation_gap=dict(gap=gap, vs_arm=gap_vs), ds0023=desc)
    tmp = dst / "analysis.json.tmp"; tmp.write_text(json.dumps(out, indent=1, default=float)); tmp.replace(dst / "analysis.json")
    md = fmt_md(out); (dst / "analysis.md").write_text(md); print(md)


def fmt_md(o):
    L = ["# EXP-0068 analysis (code/analyse.py) -- visited-state vs spawn-state offline metrics as predictors of the closed-loop order\n",
         "rho = Spearman with the arm's closed-loop score (lower-is-better metrics flipped), [state-bootstrap 95 % CI], p = exact one-sided",
         "permutation p (n = 8 models), agree = fraction of Holm-resolved closed-loop pairs ordered correctly, within-seed pairs excluded.",
         f"Models scored: {', '.join(o['models'])}.\n"]
    for a, arm in o["arms"].items():
        L.append(f"\n## arm {a} ({arm['source']}; {arm['n_episodes']} paired episodes, {arm['n_resolved']} Holm-resolved pairs)\n")
        v, s = ("visited_OTSX_matched", "spawn_OTSX") if a == "e66" else ("visited_primary", "spawn_primary")
        L.append(f"| metric | SPAWN rho [CI] | p | agree | VISITED rho [CI] | p | agree | delta rho |"); L.append("|---|---|---|---|---|---|---|---|")
        for k in METRICS:
            ts, tv = o["table"][a][s][k], o["table"][a][v][k]
            if not ts or not tv:
                continue
            f = lambda t: f"{t['rho']:+.2f} [{t['rho_ci'][0]:+.2f}, {t['rho_ci'][1]:+.2f}] | {t['perm_p']:.3f} | " + (f"{t['agree_excl_seed']:.2f} of {t['n_pairs_excl']}" if t["agree_excl_seed"] is not None else "-")
            L.append(f"| {k}{'' if METRICS[k] else ' (flipped)'} | {f(ts)} | {f(tv)} | {tv['rho'] - ts['rho']:+.2f} |")
        L.append(f"\nsecondary aggregations (rho / agree):")
        for name in o["table"][a]:
            if name in (v, s):
                continue
            L.append(f"- {name}: " + "; ".join(f"{k} {t['rho']:+.2f}" + (f" / {t['agree_excl_seed']:.2f}" if t and t['agree_excl_seed'] is not None else "")
                                              for k, t in o["table"][a][name].items() if t))
    H1, H2 = o["H1"], o["H2"]
    L.append(f"\n## H1 (visited beats spawn, rho): CEM {H1.get('cem', {}).get('n_metrics_rho_visited_higher')} / 8, rank {H1.get('rank', {}).get('n_metrics_rho_visited_higher')} / 8"
             + (f", EXP-0066 (OTSX) {H1['e66']['n_metrics_rho_visited_higher']} / 8" if "e66" in H1 else "") + f" -> **{H1['verdict']}**")
    L.append(f"pair agreement counts: CEM {H1.get('cem', {}).get('n_metrics_agree_visited_higher')} / 8, rank {H1.get('rank', {}).get('n_metrics_agree_visited_higher')} / 8")
    L.append(f"\n## H2 (accuracy_1 off chance on visited states, CEM): rho {H2['rho_visited']:+.2f} (spawn {H2['rho_spawn']:+.2f}), agree "
             f"{H2['agree_excl_seed_visited']} (spawn {H2['agree_excl_seed_spawn']}) -> **{H2['verdict']}**")
    L.append("\n## per-model values (spawn primary | visited primary) and the generalisation gap (accuracy_1 spawn - visited)\n")
    L.append("| model | " + " | ".join(METRICS) + " | gap |"); L.append("|---" * (2 + len(METRICS)) + "|")
    for m in o["models"]:
        L.append(f"| {m} | " + " | ".join(f"{o['values']['spawn_primary'][m][k]:.3f} / {o['values']['visited_primary'][m][k]:.3f}" for k in METRICS)
                 + f" | {o['generalisation_gap']['gap'][m]:+.3f} |")
    L.append("\ngap vs closed loop (lower gap = better predicted): " + "; ".join(f"{a} rho {t['rho']:+.2f} (p {t['perm_p']:.3f}, agree {t['agree_excl_seed']:.2f} of {t['n_pairs_excl']})"
                                                                                for a, t in o["generalisation_gap"]["vs_arm"].items() if t))
    d = o["ds0023"]
    if d:
        L.append(f"\n## DS-0023: {d['n_states']} states, {d['n_rows']} rows ({d['n_sampler_rows']} sampler, {d['n_planner_rows']} planner of which "
                 f"{d['n_planner_legal']} legalisable and {d['n_planner_valid_length']} exact-length); null rows sampler {d['null_rows_sampler']}, planner "
                 f"{d['null_rows_planner']}; states with a cube above layer 0: {d['n_layer1_states']}; overlapping-cube states: {d['n_illegal_states']}; "
                 f"planner candidates' mean true dv {d['planner_vs_sampler_true_dv']['mean_true_dv_planner']} vs sampler {d['planner_vs_sampler_true_dv']['mean_true_dv_sampler']}; "
                 f"planner set holds the true best in {d['planner_vs_sampler_true_dv']['frac_states_planner_holds_true_best']} of states; max settle drift {d['settle_drift_mm_max']} mm")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
