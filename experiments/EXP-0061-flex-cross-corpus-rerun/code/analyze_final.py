"""EXP-0061 final analysis: slateN tables, accuracy CIs, paired tests, strata, val-vs-test.

Reads results/final_eval/{ds0019,ds0020val}{.json,_rows.npz} (code/final_eval.py) and
writes results/final_eval/summary.json + summary.md. accuracy is ALWAYS a ratio of
population means (1 - mean(rms_model) / mean(rms_persistence)) inside whatever group is
being scored -- never a mean of per-row ratios (CODEMAP trap).

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/analyze_final.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from Baselines.common.paired_stats import paired_comparison, variance_components  # noqa: E402

R = Path("experiments/EXP-0061-flex-cross-corpus-rerun/results/final_eval")
GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
NFD = ["nfd_s0", "nfd_s1", "nfd_s2"]
GNN = ["gnn_flex_drp_samp0", "gnn_flex_drp_samp1", "gnn_flex_drp_samp2"]
LF = ["lf_flex_switched", "lf_flex_single"]
SENS = ["gnn_flex_drp_n50"]   # sensitivity rows (not in the family comparisons)
NB = 2000
rng = np.random.default_rng(0)


def acc(e_m, e_p):
    return float(1 - e_m.mean() / e_p.mean())


def boot_acc(e_m, e_p, groups=None, e_m2=None):
    """95% CI of accuracy (or of acc(m) - acc(m2) if e_m2 given): row bootstrap
    (groups=None) or cluster bootstrap over `groups` (slates / trajectories)."""
    n = len(e_p)
    out = []
    if groups is None:
        for _ in range(NB):
            i = rng.integers(0, n, n)
            a = 1 - e_m[i].mean() / e_p[i].mean()
            if e_m2 is not None:
                a -= 1 - e_m2[i].mean() / e_p[i].mean()
            out.append(a)
    else:
        ug, inv = np.unique(groups, return_inverse=True)
        G = len(ug)
        sm = np.bincount(inv, e_m, G); sp = np.bincount(inv, e_p, G); cnt = np.bincount(inv, None, G)
        sm2 = np.bincount(inv, e_m2, G) if e_m2 is not None else None
        for _ in range(NB):
            j = rng.integers(0, G, G)
            a = 1 - (sm[j].sum() / cnt[j].sum()) / (sp[j].sum() / cnt[j].sum())
            if e_m2 is not None:
                a -= 1 - (sm2[j].sum() / cnt[j].sum()) / (sp[j].sum() / cnt[j].sum())
            out.append(a)
    lo, hi = np.quantile(out, [0.025, 0.975])
    return [float(lo), float(hi)]


def per_slate_matrix(caps, name, goals=GOALS, vfs=VFS):
    """(S,) per-slate score = mean over the chosen goal x vf cells (within-state average)."""
    ps = caps[name]["capture"]["per_slate"]
    return np.mean([ps[g][v] for g in goals for v in vfs], 0)


def boot_mean_ci(x):
    i = rng.integers(0, len(x), (NB, len(x)))
    return [float(v) for v in np.quantile(x[i].mean(1), [0.025, 0.975])]


def main():
    d19 = json.load(open(R / "ds0019.json"))
    r19 = dict(np.load(R / "ds0019_rows.npz"))
    m19 = d19["models"]
    have = [m for m in NFD + GNN + LF + SENS if m in m19]
    gnn_have = [m for m in GNN if m in m19]
    S = len(m19["persistence"]["capture"]["slate_ids"])
    slate_ids = np.array(m19["persistence"]["capture"]["slate_ids"])
    out = dict(n_slates=S, n_rows=int(len(r19["slate"])), models_scored=have)

    # family per-slate (seed-mean) captures as pseudo-models
    def fam_ps(members, g, v):
        return np.mean([m19[m]["capture"]["per_slate"][g][v] for m in members], 0)
    fams = {"NFD (3 seeds)": NFD, "GNN (3 sampling seeds)": gnn_have, "lf_flex_switched": ["lf_flex_switched"],
            "lf_flex_single": ["lf_flex_single"], "random": ["random"], "persistence (DEGENERATE)": ["persistence"]}

    # ---------------- 1. slateN tables
    sl = {}
    for m in have + ["random", "persistence"]:
        ps = m19[m]["capture"]["per_slate"]
        sl[m] = {g: {v: dict(mean=float(np.mean(ps[g][v])), sem=float(np.std(ps[g][v], ddof=1) / np.sqrt(S)))
                     for v in VFS} for g in GOALS}
        sl[m]["avg_goals"] = {v: dict(mean=float(np.mean([np.mean(ps[g][v]) for g in GOALS])),
                                      sem=float(np.std(np.mean([ps[g][v] for g in GOALS], 0), ddof=1) / np.sqrt(S)))
                              for v in VFS}
    out["slateN_per_model"] = sl
    fam = {}
    for f, mem in fams.items():
        if not mem:
            continue
        fam[f] = {}
        for g in GOALS + ("avg_goals",):
            fam[f][g] = {}
            for v in VFS:
                vals = [sl[m][g][v]["mean"] for m in mem]
                xs = (np.mean([fam_ps(mem, gg, v) for gg in GOALS], 0) if g == "avg_goals" else fam_ps(mem, g, v))
                fam[f][g][v] = dict(mean=float(np.mean(vals)), seed_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                    seed_min=float(min(vals)), seed_max=float(max(vals)),
                                    slate_boot_ci=boot_mean_ci(xs))
    out["slateN_family"] = fam

    # ---------------- 2. accuracy with CIs
    ep = r19["rms_persistence"]
    A = {}
    for m in have:
        e = r19[f"rms_{m}"]
        A[m] = dict(accuracy=acc(e, ep), harness_accuracy=m19[m]["accuracy"],
                    row_boot_ci=boot_acc(e, ep), slate_cluster_ci=boot_acc(e, ep, r19["slate"]))
    A["persistence"] = dict(accuracy=acc(ep, ep))
    e_nfd = np.mean([r19[f"rms_{m}"] for m in NFD], 0)  # NOT an ensemble prediction: mean of per-row errors
    A["NFD seed-mean"] = dict(accuracy=float(np.mean([A[m]["accuracy"] for m in NFD])),
                              seed_sd=float(np.std([A[m]["accuracy"] for m in NFD], ddof=1)))
    if gnn_have:
        A["GNN sampling-seed-mean"] = dict(accuracy=float(np.mean([A[m]["accuracy"] for m in gnn_have])),
                                           seed_sd=float(np.std([A[m]["accuracy"] for m in gnn_have], ddof=1)) if len(gnn_have) > 1 else None)
    # paired accuracy deltas (row bootstrap and slate-cluster bootstrap)
    pairs = []
    ref = ["nfd_s0", "lf_flex_switched", "lf_flex_single"] + gnn_have[:1]
    for i, a in enumerate(ref):
        for b in ref[i + 1:]:
            ea, eb = r19[f"rms_{a}"], r19[f"rms_{b}"]
            pairs.append(dict(a=a, b=b, delta=acc(ea, ep) - acc(eb, ep), row_ci=boot_acc(ea, ep, None, eb),
                              slate_ci=boot_acc(ea, ep, r19["slate"], eb)))
    A["paired_deltas"] = pairs
    out["accuracy_ds0019"] = A

    # ---------------- 3. paired slateN comparisons (slate = replication unit, Holm)
    fam_models = {"NFD": NFD, "GNN": gnn_have, "LF_sw": ["lf_flex_switched"], "LF_single": ["lf_flex_single"],
                  "random": ["random"]}
    fam_models = {k: v for k, v in fam_models.items() if v}
    names = list(fam_models)
    paired = {}
    for label, vfs in [("all_9_cells", VFS)] + [(v, (v,)) for v in VFS]:
        X = np.stack([np.mean([per_slate_matrix(m19, m, GOALS, vfs) for m in mem], 0) for mem in fam_models.values()])
        paired[label] = dict(pairs=paired_comparison(X, names, n_boot=10000, seed=0),
                             variance=variance_components(X[:-1]))  # exclude random from the ANOVA
    # individual NFD seeds vs GNN seed 0 (does every seed beat it?)
    ind = NFD + gnn_have + LF
    X = np.stack([per_slate_matrix(m19, m) for m in ind])
    paired["individual_models_all_9_cells"] = dict(pairs=paired_comparison(X, ind, n_boot=10000, seed=0))
    # within-family noise floors
    paired["nfd_seed_noise"] = dict(pairs=paired_comparison(np.stack([per_slate_matrix(m19, m) for m in NFD]), NFD, seed=0))
    if len(gnn_have) > 1:
        paired["gnn_sampling_noise"] = dict(pairs=paired_comparison(np.stack([per_slate_matrix(m19, m) for m in gnn_have]),
                                                                    gnn_have, seed=0))
    # per-slate accuracy (ratio of means within slate), paired
    sid, inv = np.unique(r19["slate"], return_inverse=True)
    sp = np.bincount(inv, ep)
    Xa = np.stack([np.mean([1 - np.bincount(inv, r19[f"rms_{m}"]) / sp for m in mem], 0)
                   for k, mem in fam_models.items() if k != "random"])
    paired["accuracy_per_slate"] = dict(pairs=paired_comparison(Xa, [k for k in fam_models if k != "random"], seed=0))
    out["paired"] = paired

    # ---------------- 4. strata
    blob = r19["init_pos_blob"]
    meta = {int(k): v for k, v in d19["slate_meta"].items()}
    nrig_s = np.array([meta[int(s)]["n_rigids"] for s in slate_ids])
    t1, t2 = np.quantile(nrig_s, [1 / 3, 2 / 3])
    terc_s = np.digitize(nrig_s, [t1, t2])            # 0 small, 1 mid, 2 large
    blob_s = np.array([meta[int(s)]["init_pos"] == "rand_blob" for s in slate_ids])
    terc_row = np.digitize(r19["n_rigids"], [t1, t2])
    nvox_path = Path(__file__).parent.parent / "results/final_eval/ds0019_gnn_nvox.json"
    fb_s = None
    if nvox_path.exists():
        nv = json.load(open(nvox_path))
        fb_s = np.array([nv[str(int(s))] <= 200 for s in slate_ids])
    strata_defs = {"init_pos=rand_blob": (blob, blob_s), "init_pos=rand_spread": (~blob, ~blob_s)}
    for k, lab in enumerate(["small", "mid", "large"]):
        strata_defs[f"pieces_{lab}"] = (terc_row == k, terc_s == k)
    if fb_s is not None:
        fb_row = np.isin(r19["slate"], slate_ids[fb_s])
        strata_defs["gnn_fallback_states(<=200 voxels)"] = (fb_row, fb_s)
        strata_defs["gnn_fps_states(>200 voxels)"] = (~fb_row, ~fb_s)
    st = dict(tercile_edges_n_rigids=[float(t1), float(t2)], strata={})
    for sname, (rmask, smask) in strata_defs.items():
        e = {}
        for f, mem in fams.items():
            if not mem or f.startswith("persistence"):
                continue
            ent = dict(n_slates=int(smask.sum()), n_rows=int(rmask.sum()))
            xs = np.mean([per_slate_matrix(m19, m) for m in mem], 0)[smask]
            ent["slateN_avg9"] = float(xs.mean()); ent["slateN_ci"] = boot_mean_ci(xs)
            for v in VFS:
                ent[f"slateN_{v}"] = float(np.mean([per_slate_matrix(m19, m, GOALS, (v,)) for m in mem], 0)[smask].mean())
            if f != "random":
                accs = [acc(r19[f"rms_{m}"][rmask], ep[rmask]) for m in mem]
                ent["accuracy"] = float(np.mean(accs))
                ent["accuracy_seed_range"] = [float(min(accs)), float(max(accs))]
                ent["accuracy_slate_ci_member0"] = boot_acc(r19[f"rms_{mem[0]}"][rmask], ep[rmask], r19["slate"][rmask])
            e[f] = ent
        # paired per-slate deltas inside the stratum (avg of 9 cells; family seed-means)
        fx = {k: np.mean([per_slate_matrix(m19, m) for m in mem], 0)[smask] for k, mem in
              [("NFD", NFD), ("GNN", gnn_have), ("LF_sw", ["lf_flex_switched"]), ("LF_single", ["lf_flex_single"])] if mem}
        e["paired"] = {f"{a}-{b}": dict(mean=float((fx[a] - fx[b]).mean()), ci=boot_mean_ci(fx[a] - fx[b]))
                       for a, b in [("NFD", "GNN"), ("NFD", "LF_sw"), ("GNN", "LF_sw"), ("GNN", "LF_single")]}
        st["strata"][sname] = e
    out["strata"] = st

    # ---------------- 5. degeneracy
    out["degeneracy_frac_dv_true_zero"] = {g: {v: d19["degeneracy"][g][v]["frac_dv_true_zero"] for v in VFS} for g in GOALS}
    out["degeneracy_frac_slates_flat"] = {g: {v: d19["degeneracy"][g][v]["frac_slates_flat"] for v in VFS} for g in GOALS}

    # ---------------- 6. val vs test
    if (R / "ds0020_val.json").exists():
        d20 = json.load(open(R / "ds0020_val.json"))
        r20 = dict(np.load(R / "ds0020_val_rows.npz"))
        ep20 = r20["rms_persistence"]
        vt = {}
        for m in have:
            m20 = m if m in d20["models"] else ("gnn_flex_drp_samp0" if m.startswith("gnn_flex_drp_samp") else None)
            if m20 is None or m20 not in d20["models"] or f"rms_{m20}" not in r20:
                continue
            e20 = r20[f"rms_{m20}"]
            va = acc(e20, ep20)
            vt[m] = dict(val=va, val_traj_ci=boot_acc(e20, ep20, r20["slate"]), test=A[m]["accuracy"],
                         test_slate_ci=A[m]["slate_cluster_ci"], gap_test_minus_val=A[m]["accuracy"] - va,
                         val_from=m20)
        out["val_vs_test_accuracy"] = vt
        order_val = sorted(vt, key=lambda m: -vt[m]["val"])
        order_test = sorted(vt, key=lambda m: -vt[m]["test"])
        out["ranking_accuracy"] = dict(val=order_val, test=order_test)
    out["slateN_n_rigids_per_slate"] = sorted(int(x) for x in nrig_s)
    out["ranking_slateN_avg9_test"] = sorted(have, key=lambda m: -float(per_slate_matrix(m19, m).mean()))

    json.dump(out, open(R / "summary.tmp.json", "w"), indent=1)
    os.replace(R / "summary.tmp.json", R / "summary.json")
    write_md(out)
    print("wrote", R / "summary.json")


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def write_md(o):
    L = []
    fam = o["slateN_family"]
    L.append("## slateN averaged over 3 goals (family mean; seed range; slate-bootstrap 95% CI of the family mean)\n")
    L.append("| model | " + " | ".join(VFS) + " |")
    L.append("|---|---|---|---|")
    for f, d in fam.items():
        cells = []
        for v in VFS:
            c = d["avg_goals"][v]
            rngs = f" [{c['seed_min']:.3f}-{c['seed_max']:.3f}]" if c["seed_sd"] is not None else ""
            cells.append(f"{c['mean']:.3f}{rngs} CI {c['slate_boot_ci'][0]:.3f}..{c['slate_boot_ci'][1]:.3f}")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L.append("\n## slateN per goal x vf (per model, mean +- slate sem)\n")
    L.append("| model | " + " | ".join(f"{g}/{v}" for g in GOALS for v in VFS) + " |")
    L.append("|" + "---|" * 10)
    for m, d in o["slateN_per_model"].items():
        L.append(f"| {m} | " + " | ".join(f"{d[g][v]['mean']:.3f}+-{d[g][v]['sem']:.3f}" for g in GOALS for v in VFS) + " |")
    L.append("\n## accuracy DS-0019 (ratio of population means)\n")
    L.append("| model | accuracy | row-boot CI | slate-cluster CI |")
    L.append("|---|---|---|---|")
    for m, d in o["accuracy_ds0019"].items():
        if m == "paired_deltas":
            continue
        L.append(f"| {m} | {d['accuracy']:.4f} | {d.get('row_boot_ci', '')} | {d.get('slate_cluster_ci', d.get('seed_sd', ''))} |")
    L.append("\n### paired accuracy deltas\n")
    for p in o["accuracy_ds0019"]["paired_deltas"]:
        L.append(f"- {p['a']} - {p['b']}: {p['delta']:+.4f} row CI {p['row_ci']} slate CI {p['slate_ci']}")
    L.append("\n## paired slateN (Holm)\n")
    for lab, d in o["paired"].items():
        L.append(f"\n**{lab}**\n")
        for p in d["pairs"]:
            L.append(f"- {p['a']} vs {p['b']}: {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] "
                     f"p_holm {p['p_holm']:.4f} wins {p['wins_a']}/{p['wins_b']} resolved={p['resolved_ci']}")
    L.append("\n## strata\n")
    for s, d in o["strata"]["strata"].items():
        L.append(f"\n**{s}**\n")
        for f, e in d.items():
            if f == "paired":
                L.append("- paired: " + "; ".join(f"{k} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for k, v in e.items()))
                continue
            L.append(f"- {f}: n_slates {e['n_slates']} slateN9 {e['slateN_avg9']:.3f} CI {e['slateN_ci']} "
                     + " ".join(f"{v[:4]} {e['slateN_' + v]:.3f}" for v in VFS)
                     + (f" | acc {e['accuracy']:.3f} range {e['accuracy_seed_range']} CI0 {e['accuracy_slate_ci_member0']}" if "accuracy" in e else ""))
    L.append("\n## frac(dv_true==0)\n")
    for g, d in o["degeneracy_frac_dv_true_zero"].items():
        L.append(f"- {g}: " + ", ".join(f"{v} {x:.4f}" for v, x in d.items()))
    if "val_vs_test_accuracy" in o:
        L.append("\n## val vs test accuracy\n")
        for m, d in o["val_vs_test_accuracy"].items():
            L.append(f"- {m}: val {d['val']:.4f} {d['val_traj_ci']} test {d['test']:.4f} {d['test_slate_ci']} gap {d['gap_test_minus_val']:+.4f}")
        L.append(f"\nranking accuracy val {o['ranking_accuracy']['val']}\nranking accuracy test {o['ranking_accuracy']['test']}")
    L.append(f"\nranking slateN test {o['ranking_slateN_avg9_test']}")
    (R / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
