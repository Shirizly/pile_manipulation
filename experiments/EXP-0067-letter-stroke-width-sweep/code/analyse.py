"""EXP-0067 analysis (pre-registered in DESIGN.md; written while cell w10 ran, before any outcome was read).
Cells: base = EXP-0066 results/broad_s0_ps0.json (identical settings, config compared field by field), w10 / w15 / w20
from results/. rel_k = in-goal mass fraction at push k / the goal's mass_frac_best_placement at THAT width.
Per cell: rel at k = 8, 16 (median, q05, q95), EXP-0066 score (mean rel over pushes 4-16), completion at theta = 0.9
(max_k<=16 rel_k >= theta; Wilson 95 % CI; also k <= 8 and theta = 0.8), pushes-to-completion (censored at 16; median
from the empirical cumulative completion curve), fallback rate, executed-illegal count. H1: successive Newcombe CIs and
base->w20. H2: first width with completion > 50 %. Oracle anchors (EXP-0057 oracle_A_default base; results/oracle_w*.json).
Usage: python -u code/analyse.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import letter_goal_mask, mass_in_region  # noqa: E402
from simple_mpc.adapters import OCC_GRID, occ_for_scoring  # noqa: E402

HERE = REPO / "experiments/EXP-0067-letter-stroke-width-sweep"; R = HERE / "results"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
VS.update(json.loads((R / "vstar_width.json").read_text()))
CAL = json.loads((R / "width_calibration.json").read_text())
BASE = REPO / "experiments/EXP-0066-closed-loop-power-study/results/broad_s0_ps0.json"
LETTERS = ["O", "T", "S", "X"]; WIDTHS = ["base", "w10", "w15", "w20"]; K0, K1 = 4, 16; THETA = 0.9
IGNORE = {"tag", "results_dir", "goals"}


def opt(goal):
    return VS[goal]["mass_frac_best_placement"]


def rel_curve(states, goal):
    S = torch.tensor(states).float(); occ = occ_for_scoring(S)
    mf = (mass_in_region(occ, torch.from_numpy(letter_goal_mask(goal, OCC_GRID, OCC_GRID))) / occ.reshape(len(S), -1).sum(1)).numpy()
    return mf / opt(goal)


def load_cell(path):
    if not path.exists():
        return {}, None
    d = json.loads(path.read_text()); out = {}
    for e in d["episodes"]:
        if not e.get("complete") or not e.get("states"):
            continue
        rel = rel_curve(e["states"], e["goal"]); L = e["goal"].split("_")[1]
        first = [k for k in range(len(rel)) if rel[k] >= THETA]
        out[(L, e["start"])] = dict(rel=rel.tolist(), score=float(rel[K0:K1 + 1].mean()), rel0=float(rel[0]), rel8=float(rel[8]), rel16=float(rel[K1]),
                                    max16=float(rel[:K1 + 1].max()), max8=float(rel[:9].max()), k_done=(first[0] if first else None),
                                    n_illegal=sum(1 for o in e.get("legal_ok", []) if not o), n_fallback=int(sum(e.get("legal_fallback", []))),
                                    n_shift=sum(1 for s in e.get("legal_shift_m", []) if s > 0), n_steps=len(e["actions"]), wall_s=e.get("wall_s", 0.0))
    return out, d["config"]


def wilson(x, n, z=1.96):
    if n == 0:
        return [None, None]
    p = x / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def newcombe(x1, n1, x2, n2):
    """CI for p2 - p1 (Newcombe hybrid score, independent proportions)."""
    l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2); p1, p2 = x1 / n1, x2 / n2
    return [float(p2 - p1 - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)), float(p2 - p1 + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))]


def q(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), q05=float(np.percentile(x, 5)), q95=float(np.percentile(x, 95)), mean=float(x.mean()))


def summarise(v):
    n = len(v); done = [e for e in v.values() if e["max16"] >= THETA]; done8 = [e for e in v.values() if e["max8"] >= THETA]
    ks = sorted(e["k_done"] for e in done)
    cum = [sum(1 for k in ks if k <= t) / n for t in range(K1 + 1)]
    med_k = next((t for t in range(K1 + 1) if cum[t] >= 0.5), None)
    return dict(n=n, rel0=q([e["rel0"] for e in v.values()]), rel8=q([e["rel8"] for e in v.values()]), rel16=q([e["rel16"] for e in v.values()]),
                score=q([e["score"] for e in v.values()]), completion=len(done) / n, completion_ci=wilson(len(done), n), n_complete=len(done),
                completion_k8=len(done8) / n, completion_k8_ci=wilson(len(done8), n),
                completion_theta08=sum(1 for e in v.values() if e["max16"] >= 0.8) / n,
                pushes_to_completion=dict(median=(med_k if med_k is not None else ">16"), cumulative_by_k=cum, completed_ks=ks),
                executed_illegal=int(sum(e["n_illegal"] for e in v.values())), fallback_rate=float(sum(e["n_fallback"] for e in v.values()) / sum(e["n_steps"] for e in v.values())),
                shift_rate=float(sum(e["n_shift"] for e in v.values()) / sum(e["n_steps"] for e in v.values())), wall_min=float(sum(e["wall_s"] for e in v.values()) / 32 / 60))


def oracle_cell(path, width):
    if not path.exists():
        return None
    d = json.loads(path.read_text()); out = {}
    for e in d["episodes"]:
        if not e["goal"].startswith(("letter_O", "letter_T")):
            continue
        g = e["goal"]
        if width == "base" and ("_w" in g):
            continue
        if width != "base" and not g.endswith("_" + width):
            continue
        rel = (np.asarray(e["mass_frac"], float) / opt(g)).tolist()
        if len(rel) < K1 + 1:
            continue
        first = [k for k in range(K1 + 1) if rel[k] >= THETA]
        out[f"{g}/{e['start']}"] = dict(rel0=rel[0], rel8=rel[8], rel16=rel[K1], max16=max(rel[:K1 + 1]), k_done=(first[0] if first else None), solved_at=e.get("solved_at"), n_pushes_recorded=len(e["actions"]))
    if not out:
        return None
    n = len(out); c = sum(1 for e in out.values() if e["max16"] >= THETA)
    return dict(tasks=out, n=n, completion=c / n, completion_ci=wilson(c, n), rel8_mean=float(np.mean([e["rel8"] for e in out.values()])), rel16_mean=float(np.mean([e["rel16"] for e in out.values()])), step_recorded=d.get("step"))


def main():
    cells, cfgs = {}, {}
    cells["base"], cfgs["base"] = load_cell(BASE)
    for w in WIDTHS[1:]:
        cells[w], cfgs[w] = load_cell(R / f"{w}.json")
    cells = {w: v for w, v in cells.items() if v}
    ref = next((cfgs[w] for w in WIDTHS[1:] if cfgs.get(w)), None)
    cfg_diff = {k: (cfgs["base"].get(k), ref.get(k)) for k in set(cfgs["base"]) | set(ref) if k not in IGNORE and cfgs["base"].get(k) != ref.get(k)} if ref and cfgs.get("base") else "n/a"
    res = dict(cells_present=[w for w in WIDTHS if w in cells], cells_missing=[w for w in WIDTHS if w not in cells], base_config_diff_vs_sweep=cfg_diff,
               realised_width_mm={w: ({"base": CAL["baseline"], "w10": CAL["realised"]["10"], "w15": CAL["realised"]["15"], "w20": CAL["realised"]["20"]}[w]) for w in WIDTHS},
               optimum={w: {L: opt(f"letter_{L}" + ("" if w == "base" else "_" + w)) for L in LETTERS} for w in WIDTHS}, per_cell={}, per_cell_letter={})
    for w, v in cells.items():
        res["per_cell"][w] = summarise(v)
        res["per_cell_letter"][w] = {L: summarise({k: e for k, e in v.items() if k[0] == L}) for L in LETTERS if any(k[0] == L for k in v)}
    # H1: successive and base->w20 completion differences (Newcombe), monotonicity
    pres = res["cells_present"]; diffs = []
    for a, b in zip(pres[:-1], pres[1:]):
        A, B = res["per_cell"][a], res["per_cell"][b]
        diffs.append(dict(a=a, b=b, diff=B["completion"] - A["completion"], ci=newcombe(A["n_complete"], A["n"], B["n_complete"], B["n"])))
    res["H1"] = dict(successive=diffs, monotone=all(d["diff"] >= 0 for d in diffs))
    if "base" in cells and "w20" in cells:
        A, B = res["per_cell"]["base"], res["per_cell"]["w20"]
        res["H1"]["base_to_w20"] = dict(diff=B["completion"] - A["completion"], ci=newcombe(A["n_complete"], A["n"], B["n_complete"], B["n"]))
        res["H1"]["supported"] = bool(res["H1"]["monotone"] and res["H1"]["base_to_w20"]["ci"][0] > 0)
    first = next((w for w in pres if res["per_cell"][w]["completion"] > 0.5), None)
    res["H2"] = dict(first_width_over_50pct=first, supported=(first in ("w10", "w15", "w20")) if first or "w20" in cells else None,
                     embarrassing_under_10pct_at_w20=(res["per_cell"]["w20"]["completion"] < 0.10) if "w20" in cells else None,
                     opposite_w10_over_50pct=(res["per_cell"]["w10"]["completion"] > 0.5) if "w10" in cells else None)
    res["oracle"] = {"base": oracle_cell(REPO / "experiments/EXP-0057-oracle-budget-ablation/results/oracle_A_default.json", "base"),
                     "w15": oracle_cell(R / "oracle_w15.json", "w15"), "w10": oracle_cell(R / "oracle_w10.json", "w10")}
    tmp = R / "analysis.json.tmp"; tmp.write_text(json.dumps(res, indent=1)); tmp.replace(R / "analysis.json")
    L = ["# EXP-0067 analysis (generated by code/analyse.py)\n", f"cells present: {res['cells_present']}; missing: {res['cells_missing'] or 'none'}; base-cell config differences vs sweep (excluding tag/results_dir/goals): {cfg_diff or 'none'}\n",
         "| width | realised median stroke mm (O/T/S/X) | optimum (O/T/S/X) | n | completion @0.9 [Wilson] | @0.9 by k<=8 | @0.8 | pushes-to-completion median | rel8 median [q05, q95] | rel16 median [q05, q95] | score (4-16) mean | rel0 median | illegal | fallback | shift |",
         "|---" * 15 + "|"]
    for w in pres:
        c = res["per_cell"][w]; rw = res["realised_width_mm"][w]; o = res["optimum"][w]
        rws = '/'.join(f"{rw[l]['median_mm']:.1f}" for l in LETTERS); os_ = '/'.join(f"{o[l]:.3f}" for l in LETTERS)
        L.append(f"| {w} | {rws} | {os_} | {c['n']} | {c['completion']:.3f} [{c['completion_ci'][0]:.3f}, {c['completion_ci'][1]:.3f}] | {c['completion_k8']:.3f} | {c['completion_theta08']:.3f} | {c['pushes_to_completion']['median']} | {c['rel8']['median']:.3f} [{c['rel8']['q05']:.3f}, {c['rel8']['q95']:.3f}] | {c['rel16']['median']:.3f} [{c['rel16']['q05']:.3f}, {c['rel16']['q95']:.3f}] | {c['score']['mean']:.3f} | {c['rel0']['median']:.3f} | {c['executed_illegal']} | {c['fallback_rate']:.3f} | {c['shift_rate']:.2f} |")
    L.append("\n| width | " + " | ".join(f"{l}: completion / rel16 median" for l in LETTERS) + " |\n|---|---|---|---|---|")
    for w in pres:
        L.append(f"| {w} | " + " | ".join(f"{res['per_cell_letter'][w][l]['completion']:.2f} / {res['per_cell_letter'][w][l]['rel16']['median']:.3f}" if l in res["per_cell_letter"][w] else "-" for l in LETTERS) + " |")
    h = res["H1"]; L.append("\n## H1 (monotone completion in width)\n" + "; ".join(f"{d['a']}->{d['b']} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" for d in h["successive"]) + f"; monotone: {h['monotone']}" + (f"; base->w20 {h['base_to_w20']['diff']:+.3f} [{h['base_to_w20']['ci'][0]:+.3f}, {h['base_to_w20']['ci'][1]:+.3f}]; supported: {h['supported']}" if "base_to_w20" in h else ""))
    L.append(f"\n## H2\nfirst width with completion > 50 %: {res['H2']['first_width_over_50pct']}; supported: {res['H2']['supported']}; embarrassing (< 10 % at w20): {res['H2']['embarrassing_under_10pct_at_w20']}; opposite (w10 > 50 %): {res['H2']['opposite_w10_over_50pct']}")
    L.append("\n## Oracle anchor (perfect-model CEM 64x4, 256 sims/push; O and T)")
    for w, o in res["oracle"].items():
        if o is None:
            L.append(f"- {w}: not run"); continue
        L.append(f"- {w}: n {o['n']} tasks, completion {o['completion']:.2f} [{o['completion_ci'][0]:.2f}, {o['completion_ci'][1]:.2f}], rel8 mean {o['rel8_mean']:.3f}, rel16 mean {o['rel16_mean']:.3f}; per task " + "; ".join(f"{k} rel16 {t['rel16']:.3f}" + (f" done@{t['k_done']}" if t['k_done'] is not None else "") for k, t in o["tasks"].items()))
    (R / "analysis.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
