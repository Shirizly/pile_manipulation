"""EXP-0066 analysis (pre-registered in DESIGN.md). Reads results/<tag>.json cells, writes results/analysis.{json,md}.
Score per episode = mean over pushes 4..16 of in-goal mass fraction / the goal's mass_frac_best_placement (EXP-0046),
the EXP-0039 RUN-0002 formula. H1 paired bootstrap of the family gap, H1h hierarchical bootstrap (seeds / instances
resampled too), H2 spreads, MDG, variance decomposition, Holm pairs, executed-illegal and fallback counts.
Usage: python code/analyse.py"""
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask  # noqa: E402
from Baselines.common.goals import mass_in_region  # noqa: E402
from simple_mpc.adapters import occ_for_scoring  # noqa: E402

HERE = REPO / "experiments/EXP-0066-closed-loop-power-study"; R = HERE / "results"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
K0, K1 = 4, 16
rng = np.random.default_rng(0)
NFD = ["nfd_s0_ps0", "nfd_s1_ps0", "nfd_s2_ps0"]; LIN = ["lin_orig_ps0", "lin_b1_ps0", "lin_b2_ps0"]
PS_NFD = ["nfd_s0_ps0", "nfd_s0_ps1", "nfd_s0_ps2"]; PS_LIN = ["lin_orig_ps0", "lin_orig_ps1", "lin_orig_ps2"]
PRIMARY = NFD + LIN + PS_NFD[1:] + PS_LIN[1:]; OPTIONAL = ["broad_s0_ps0", "broad_s1_ps0"]
FAMILY = {t: ("nfd" if t.startswith("nfd") else "lin" if t.startswith("lin") else "broad") for t in PRIMARY + OPTIONAL}
INSTANCE = {t: t.rsplit("_ps", 1)[0] for t in PRIMARY + OPTIONAL}


def load_cell(tag):
    f = R / f"{tag}.json"
    if not f.exists():
        return {}
    out = {}
    for e in json.loads(f.read_text())["episodes"]:
        if not e.get("complete") or not e.get("states"):
            continue
        S = torch.tensor(e["states"]).float(); occ = occ_for_scoring(S)
        mf = (mass_in_region(occ, torch.from_numpy(goal_mask(e["goal"]))) / occ.reshape(len(S), -1).sum(1)).numpy()
        rel = mf / VS[e["goal"]]["mass_frac_best_placement"]
        out[(e["goal"], e["start"])] = dict(score=float(rel[K0:K1 + 1].mean()), k0=float(rel[0]), k16=float(rel[K1]),
                                            n_illegal=sum(1 for o in e.get("legal_ok", []) if not o),
                                            n_fallback=int(sum(e.get("legal_fallback", []))),
                                            n_shift=sum(1 for s in e.get("legal_shift_m", []) if s > 0),
                                            n_steps=len(e["actions"]), wall_s=e.get("wall_s", 0.0))
    return out


def q(x):
    x = np.asarray(x, float); bs = rng.choice(x, (4000, len(x))).mean(1)
    return dict(n=int(len(x)), mean=float(x.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                median=float(np.median(x)), q05=float(np.percentile(x, 5)), q95=float(np.percentile(x, 95)), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def paired(d, B=10000):
    d = np.asarray(d, float); bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                p=float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))), paired_sd=float(d.std(ddof=1)), n=int(len(d)),
                se=float(d.std(ddof=1) / np.sqrt(len(d))))


def main():
    cells = {t: load_cell(t) for t in PRIMARY + OPTIONAL}
    cells = {t: v for t, v in cells.items() if v}
    res = dict(cells_present=sorted(cells), cells_missing=[t for t in PRIMARY if t not in cells],
               optional_missing=[t for t in OPTIONAL if t not in cells], per_cell={}, per_cell_goal={})
    for t, v in cells.items():
        sc = [e["score"] for e in v.values()]
        res["per_cell"][t] = dict(**q(sc), k0=float(np.mean([e["k0"] for e in v.values()])), k16=float(np.mean([e["k16"] for e in v.values()])),
                                  executed_illegal=int(sum(e["n_illegal"] for e in v.values())),
                                  fallback_rate=float(sum(e["n_fallback"] for e in v.values()) / sum(e["n_steps"] for e in v.values())),
                                  shift_rate=float(sum(e["n_shift"] for e in v.values()) / sum(e["n_steps"] for e in v.values())),
                                  wall_min=float(sum(e["wall_s"] for e in v.values()) / 32 / 60))
        res["per_cell_goal"][t] = {g: q([e["score"] for k, e in v.items() if k[0] == g]) for g in sorted({k[0] for k in v})}
    # H1 / H1h on the episodes common to the available ps0 cells of both families
    nfd = [t for t in NFD if t in cells]; lin = [t for t in LIN if t in cells]
    if nfd and lin:
        keys = sorted(set.intersection(*[set(cells[t]) for t in nfd + lin]))
        Xn = np.array([[cells[t][k]["score"] for k in keys] for t in nfd]); Xl = np.array([[cells[t][k]["score"] for k in keys] for t in lin])
        d = Xn.mean(0) - Xl.mean(0)
        res["H1"] = dict(nfd_cells=nfd, lin_cells=lin, **paired(d))
        res["H1"]["mdg_seeds_fixed"] = float(2.8 * res["H1"]["se"])
        if "lin_orig_ps0" in cells:
            res["H1_full_fit_linear_only"] = paired(Xn.mean(0) - np.array([cells["lin_orig_ps0"][k]["score"] for k in keys]))
        B = 10000; gaps = np.empty(B)
        for b in range(B):
            ei = rng.integers(0, len(keys), len(keys)); si = rng.integers(0, len(nfd), len(nfd)); li = rng.integers(0, len(lin), len(lin))
            gaps[b] = Xn[si][:, ei].mean() - Xl[li][:, ei].mean()
        res["H1h"] = dict(mean=float(d.mean()), ci=[float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                          p=float(min(1.0, 2 * min((gaps <= 0).mean(), (gaps >= 0).mean()))), note="episodes, NFD seeds and linear instances all resampled with replacement")
        per_goal = {}
        for g in sorted({k[0] for k in keys}):
            ii = [i for i, k in enumerate(keys) if k[0] == g]
            per_goal[g] = paired(d[ii])
        res["H1_per_goal"] = per_goal
        # H2 spreads
        G = abs(d.mean())
        def spread(tags):
            tags = [t for t in tags if t in cells]
            m = [res["per_cell"][t]["mean"] for t in tags]
            sd = float(np.std(m, ddof=1)) if len(m) > 1 else None
            return dict(cells=tags, means=m, sd=sd, sd_over_gap=(sd / G if sd is not None and G > 0 else None))
        res["H2"] = dict(training_seed=spread(NFD), linear_instance=spread(LIN), planner_seed_nfd=spread(PS_NFD), planner_seed_lin=spread(PS_LIN), gap=float(d.mean()))
        sds = res["H2"]
        if sds["training_seed"]["sd"] is not None and sds["linear_instance"]["sd"] is not None:
            se_r = np.sqrt(sds["training_seed"]["sd"] ** 2 / len(nfd) + sds["linear_instance"]["sd"] ** 2 / len(lin))
            res["H1"]["mdg_seeds_random"] = float(2.8 * se_r); res["H1"]["se_seeds_random"] = float(se_r)
        # paired cell comparisons among the ps0 cells, Holm; and the planner-replicate pairs
        def pairs(tags):
            tags = [t for t in tags if t in cells]; out = []
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    kk = sorted(set(cells[tags[i]]) & set(cells[tags[j]]))
                    pr = paired([cells[tags[i]][k]["score"] - cells[tags[j]][k]["score"] for k in kk])
                    out.append(dict(a=tags[i], b=tags[j], **pr))
            order = sorted(range(len(out)), key=lambda t: out[t]["p"]); still = True
            for r, t in enumerate(order):
                still = still and out[t]["p"] * (len(out) - r) < 0.05; out[t]["holm_resolved"] = bool(still)
            return out
        res["pairs_ps0"] = pairs(NFD + LIN); res["pairs_planner_replicates"] = pairs(PS_NFD) + pairs(PS_LIN)
        # variance decomposition (sequential SS) over every primary cell's episodes
        rows = [(t, k[0], k[1], e["score"]) for t in PRIMARY if t in cells for k, e in cells[t].items()]
        y = np.array([r[3] for r in rows]); y0 = y - y.mean()
        def dummies(vals):
            u = sorted(set(vals)); return np.array([[1.0 if v == a else 0.0 for a in u[1:]] for v in vals])
        blocks = [("goal", dummies([r[1] for r in rows])), ("start_within_goal", dummies([(r[1], r[2]) for r in rows])),
                  ("family", dummies([FAMILY[r[0]] for r in rows])), ("instance_within_family", dummies([INSTANCE[r[0]] for r in rows])),
                  ("planner_seed_within_instance", dummies([r[0] for r in rows]))]
        X = np.ones((len(rows), 1)); sst = float((y0 ** 2).sum()); prev = sst; vd = {}; rank = 1
        for name, Dm in blocks:
            X = np.hstack([X, Dm]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); ssr = float(((y - X @ beta) ** 2).sum())
            r = int(np.linalg.matrix_rank(X)); vd[name] = dict(ss_fraction=(prev - ssr) / sst, df=r - rank); prev, rank = ssr, r
        vd["residual"] = dict(ss_fraction=prev / sst, df=int(len(rows) - rank))
        res["variance_decomposition"] = dict(n_rows=len(rows), total_ss=sst, components=vd, note="sequential sums of squares in the listed order; planner-seed cells only exist for 2 instances")
    tmp = R / "analysis.json.tmp"; tmp.write_text(json.dumps(res, indent=1)); tmp.replace(R / "analysis.json")
    L = ["# EXP-0066 analysis (generated by code/analyse.py)\n", f"cells present: {', '.join(res['cells_present'])}; missing primary: {res['cells_missing'] or 'none'}; optional missing: {res['optional_missing'] or 'none'}\n",
         "| cell | n | mean [CI] | median | q05 | q95 | sd | k0 | k16 | executed illegal | fallback rate | shift rate | min/cell |", "|---" * 13 + "|"]
    for t, v in res["per_cell"].items():
        L.append(f"| {t} | {v['n']} | {v['mean']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['median']:.3f} | {v['q05']:.3f} | {v['q95']:.3f} | {v['sd']:.3f} | {v['k0']:.3f} | {v['k16']:.3f} | {v['executed_illegal']} | {v['fallback_rate']:.3f} | {v['shift_rate']:.2f} | {v['wall_min']:.0f} |")
    L.append("\n| cell | " + " | ".join(f"{g} median [q05, q95]" for g in ("letter_O", "letter_T", "letter_S", "letter_X")) + " |\n|---|---|---|---|---|")
    for t, v in res["per_cell_goal"].items():
        L.append(f"| {t} | " + " | ".join(f"{v[g]['median']:.3f} [{v[g]['q05']:.3f}, {v[g]['q95']:.3f}]" if g in v else "-" for g in ("letter_O", "letter_T", "letter_S", "letter_X")) + " |")
    if "H1" in res:
        h = res["H1"]; L.append(f"\n## H1 family gap (NFD {len(h['nfd_cells'])}-seed mean - linear {len(h['lin_cells'])}-instance mean), {h['n']} paired episodes\n"
                                f"G = {h['mean']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] (paired bootstrap, p {h['p']:.4f}); paired sd {h['paired_sd']:.3f}, SE {h['se']:.4f}; "
                                f"MDG seeds fixed {h['mdg_seeds_fixed']:.3f}" + (f"; MDG seeds random {h['mdg_seeds_random']:.3f} (SE {h['se_seeds_random']:.4f})" if "mdg_seeds_random" in h else ""))
        hh = res["H1h"]; L.append(f"H1h hierarchical (seeds / instances resampled): [{hh['ci'][0]:+.4f}, {hh['ci'][1]:+.4f}], p {hh['p']:.4f}")
        if "H1_full_fit_linear_only" in res:
            f = res["H1_full_fit_linear_only"]; L.append(f"NFD 3-seed mean - full-fit linear only: {f['mean']:+.4f} [{f['ci'][0]:+.4f}, {f['ci'][1]:+.4f}], p {f['p']:.4f}")
        L.append("per goal: " + "; ".join(f"{g} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for g, v in res["H1_per_goal"].items()))
        L.append("\n## H2 spreads (sd of cell means; fraction of |G|)")
        for k, v in res["H2"].items():
            if isinstance(v, dict):
                L.append(f"- {k}: cells {v['cells']} means {[round(m, 3) for m in v['means']]} sd {v['sd'] if v['sd'] is None else round(v['sd'], 4)} = {v['sd_over_gap'] if v['sd_over_gap'] is None else round(v['sd_over_gap'], 2)} x |G|")
        L.append("\n## Holm over the 15 ps0 cell pairs\n| a | b | diff [CI] | p | resolved |\n|---|---|---|---|---|")
        for p_ in res["pairs_ps0"]:
            L.append(f"| {p_['a']} | {p_['b']} | {p_['mean']:+.3f} [{p_['ci'][0]:+.3f}, {p_['ci'][1]:+.3f}] | {p_['p']:.4f} | {p_['holm_resolved']} |")
        L.append("\nplanner-replicate pairs: " + "; ".join(f"{p_['a']} vs {p_['b']} {p_['mean']:+.3f} [{p_['ci'][0]:+.3f}, {p_['ci'][1]:+.3f}] p {p_['p']:.3f}" for p_ in res["pairs_planner_replicates"]))
        vd = res["variance_decomposition"]; L.append(f"\n## Variance decomposition ({vd['n_rows']} episodes, sequential SS)\n| component | df | SS fraction |\n|---|---|---|")
        for k, v in vd["components"].items():
            L.append(f"| {k} | {v['df']} | {v['ss_fraction']:.3f} |")
    (R / "analysis.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
