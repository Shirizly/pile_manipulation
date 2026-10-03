"""EXP-0062 RUN-0002 -- score the DS-0020-v2-fit linear visual foresight (MODEL-0009,
switched + single) on DS-0019 and DS-0020 v2 val; paired slateN vs EXP-0061's v1 LF
(MODEL-0004, reused per-slate file, verified by re-scoring) and vs v2 NFD (MODEL-0008,
EXP-0062 RUN-0001 per-slate file).

Same code path and flags as RUN-0001 (`eval_nfd_v2.py`) and EXP-0061 `final_eval.py`:
corpus `flex_ds0019_mask`, truth = cell.occ1 = BINARY image mask of the true after-state
(= eval_report `--truth-scoring image`, `_capture_report(cell, pred, "default", None)`),
goals random_quadrant/ring_O/T x lyapunov/mass_in_region/signed_mass, swept-region
accuracy from the per-row rms (ratio of population means).

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_lf_v2.py score
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_lf_v2.py analyze

Outputs (atomic): results/lf_ds0019_raw.json + _raw_rows.npz, results/lf_ds0020v2_val_rows.npz
(score); results/lf_ds0019.{json,md} (analyze).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

E61 = Path("experiments/EXP-0061-flex-cross-corpus-rerun")
R61 = E61 / "results/final_eval"
R = Path("experiments/EXP-0062-flex-v2-train-rerun/results")
W9 = "weights/MODEL-0009-linear-foresight-flex-mask-v2"
D20V2 = "datasets/DS-0020-training-data-flex-N864/config.yaml"
GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
NB = 2000
# name -> (eval_report MODELS key, ckpt override or None = the spec's own = MODEL-0004)
MODELS = {
    "lf_v2_switched": ("lf_flex_switched", f"{W9}/checkpoint.pt"),
    "lf_v2_single": ("lf_flex_single", f"{W9}/checkpoint.pt"),
    "lf_v2_switched_collection_bins": ("lf_flex_switched", f"{W9}/alt_collection_bins/checkpoint.pt"),
    "lf_v1_switched_rescored": ("lf_flex_switched", None),
    "lf_v1_single_rescored": ("lf_flex_single", None),
}


def _save(tag, res, rows=None):
    from Baselines.common import eval_report as er
    R.mkdir(parents=True, exist_ok=True)
    er._write_json_atomic(str(R / f"{tag}.json"), res)
    if rows is not None:
        tmp = R / f"{tag}_rows.tmp.npz"
        np.savez(tmp, **rows)
        os.replace(tmp, R / f"{tag}_rows.npz")


def spec_of(name):
    from Baselines.common import eval_report as er
    key, ck = MODELS[name]
    spec = dict(er.MODELS[key])
    if ck:
        spec["ckpt"] = ck
    return spec


def score(args):
    import torch
    from Baselines.common import eval_report as er
    from FlexData.dataset import load_flex_cell
    from fit_linear_foresight import metrics
    from Baselines.LinearForesight.model import bin_index
    sys.path.insert(0, str(E61 / "code"))
    from final_eval import region_of, row_rms

    res = dict(truth="binary image mask of the true after-state (cell.occ1, occ_source=image_mask); "
                     "eval_report._capture_report(truth_s0=None) == --truth-scoring image",
               models={}, val={}, bin_counts_ds0019={})
    t0 = time.time()
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    o1 = cell.occ1
    assert set(torch.unique(o1).tolist()) <= {0.0, 1.0}, "truth is not a binary mask"
    res["truth_check"] = dict(unique_values=sorted(torch.unique(o1).tolist()), dtype=str(o1.dtype))
    region = region_of(cell)
    truth, prev = cell.occ1.float(), cell.occ0.float()
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    rows = dict(slate=cell.slate_idx.numpy(), push_length=cell.push_length.numpy(),
                rms_persistence=row_rms(prev, truth, region).numpy())
    assert np.array_equal(rows["slate"], r61["slate"]), "row order differs from EXP-0061"
    assert np.allclose(rows["rms_persistence"], r61["rms_persistence"]), "persistence differs from EXP-0061"
    for sch, p in [("equal", f"{W9}/checkpoint.pt"), ("collection", f"{W9}/alt_collection_bins/checkpoint.pt"),
                   ("v1_MODEL-0004", "weights/MODEL-0004-linear-foresight-flex-mask/checkpoint.pt")]:
        ed = torch.load(p, map_location="cpu", weights_only=False)["bin_edges"]
        b = bin_index(cell.push_length, ed)
        res["bin_counts_ds0019"][sch] = dict(edges=[float(e) for e in ed],
                                             counts=[int((b == k).sum()) for k in range(len(ed) - 1)])
    print(f"[ds0019] {len(cell.occ0)} rows ({time.time() - t0:.0f}s); bins {res['bin_counts_ds0019']}", flush=True)
    for name in MODELS:
        t1 = time.time()
        spec = spec_of(name)
        acc, pred = er._accuracy(spec, er._load_predictor(spec), cell, "cuda")
        rows[f"rms_{name}"] = row_rms(pred.float(), truth, region).numpy()
        cap = er._capture_report(cell, pred, "default", None)
        res["models"][name] = dict(ckpt=spec["ckpt"], factory=spec["factory"], accuracy=acc, capture=cap,
                                   seconds=time.time() - t1)
        print(name, f"acc {acc:.4f}", {k: round(v, 4) for k, v in cap["averaged_over_goals"].items()}, flush=True)
        _save("lf_ds0019_raw", res, rows)
        del pred
    # DS-0020 v2 val
    vcell = load_flex_cell(D20V2, "val", tag="ds0020v2_val_mask", occ_source="image_mask")
    vregion = region_of(vcell)
    vt, vp = vcell.occ1.float(), vcell.occ0.float()
    vrows = dict(traj=vcell.slate_idx.numpy(), push_length=vcell.push_length.numpy(),
                 rms_persistence=row_rms(vp, vt, vregion).numpy())
    v62 = dict(np.load(R / "ds0020v2_val_rows.npz"))
    assert np.array_equal(vrows["traj"], v62["traj"]) and np.allclose(vrows["rms_persistence"], v62["rms_persistence"])
    res["val"]["n_rows"] = int(len(vcell.occ0))
    res["val"]["persistence_accuracy"] = metrics(vp, vt, vp, region=vregion)["accuracy"]
    for name in MODELS:
        spec = spec_of(name)
        acc, pred = er._accuracy(spec, er._load_predictor(spec), vcell, "cuda")
        vrows[f"rms_{name}"] = row_rms(pred.float(), vt, vregion).numpy()
        res["val"][name] = dict(accuracy=acc)
        print("val", name, f"acc {acc:.4f}", flush=True)
    _save("lf_ds0019_raw", res, rows)
    tmp = R / "lf_ds0020v2_val_rows.tmp.npz"
    np.savez(tmp, **vrows)
    os.replace(tmp, R / "lf_ds0020v2_val_rows.npz")


# ------------------------------------------------------------------ analysis
rng = np.random.default_rng(0)


def acc(e, ep):
    return float(1 - e.mean() / ep.mean())


def boot_acc(e, ep, groups, e2=None):
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    sm, sp = np.bincount(inv, e, G), np.bincount(inv, ep, G)
    sm2 = np.bincount(inv, e2, G) if e2 is not None else None
    out = []
    for _ in range(NB):
        j = rng.integers(0, G, G)
        a = 1 - sm[j].sum() / sp[j].sum()
        if e2 is not None:
            a -= 1 - sm2[j].sum() / sp[j].sum()
        out.append(a)
    return [float(x) for x in np.quantile(out, [0.025, 0.975])]


def boot_mean_ci(x):
    i = rng.integers(0, len(x), (NB, len(x)))
    return [float(v) for v in np.quantile(x[i].mean(1), [0.025, 0.975])]


def ps_mat(cap, goals=GOALS, vfs=VFS):
    return np.mean([cap["per_slate"][g][v] for g in goals for v in vfs], 0)


def analyze(args):
    from Baselines.common.paired_stats import paired_comparison
    raw = json.load(open(R / "lf_ds0019_raw.json"))
    rows = dict(np.load(R / "lf_ds0019_raw_rows.npz"))
    d61 = json.load(open(R61 / "ds0019.json"))
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    nraw = json.load(open(R / "nfd_seed0_ds0019_raw.json"))
    nrows = dict(np.load(R / "nfd_seed0_ds0019_raw_rows.npz"))
    assert np.array_equal(nrows["slate"], rows["slate"])
    M = raw["models"]
    caps = {"lf_v2_switched": M["lf_v2_switched"]["capture"], "lf_v2_single": M["lf_v2_single"]["capture"],
            "lf_v2_switched_collection_bins": M["lf_v2_switched_collection_bins"]["capture"],
            "lf_v1_switched (MODEL-0004, EXP-0061 file)": d61["models"]["lf_flex_switched"]["capture"],
            "lf_v1_single (MODEL-0004, EXP-0061 file)": d61["models"]["lf_flex_single"]["capture"],
            "nfd_v2_s0 (MODEL-0008, RUN-0001 file)": nraw["models"]["nfd_v2_s0"]["capture"],
            "random": d61["models"]["random"]["capture"], "persistence": d61["models"]["persistence"]["capture"]}
    sids = caps["lf_v2_switched"]["slate_ids"]
    for c in caps.values():
        assert c["slate_ids"] == sids
    err = {"lf_v2_switched": rows["rms_lf_v2_switched"], "lf_v2_single": rows["rms_lf_v2_single"],
           "lf_v2_switched_collection_bins": rows["rms_lf_v2_switched_collection_bins"],
           "lf_v1_switched": r61["rms_lf_flex_switched"], "lf_v1_single": r61["rms_lf_flex_single"],
           "nfd_v2_s0": nrows["rms_nfd_v2_s0"]}
    # reproduction check of the reused EXP-0061 MODEL-0004 numbers
    repro = {}
    for new, old in [("lf_v1_switched_rescored", "lf_flex_switched"), ("lf_v1_single_rescored", "lf_flex_single")]:
        co, cn = d61["models"][old]["capture"], M[new]["capture"]
        repro[old] = dict(accuracy_old=d61["models"][old]["accuracy"], accuracy_new=M[new]["accuracy"],
                          max_abs_per_slate_diff=float(max(np.abs(np.array(co["per_slate"][g][v]) -
                                                                  np.array(cn["per_slate"][g][v])).max()
                                                           for g in GOALS for v in VFS)),
                          max_abs_row_rms_diff=float(np.abs(r61[f"rms_{old}"] - rows[f"rms_{new}"]).max()))
    S = len(sids)
    out = dict(n_slates=S, n_rows=int(len(rows["slate"])), truth=raw["truth"], truth_check=raw["truth_check"],
               reuse_check_MODEL_0004=repro, bin_counts_ds0019=raw["bin_counts_ds0019"],
               ckpts={k: v["ckpt"] for k, v in M.items()})
    sl = {}
    for m, cap in caps.items():
        ps = cap["per_slate"]
        sl[m] = {g: {v: dict(mean=float(np.mean(ps[g][v])), sem=float(np.std(ps[g][v], ddof=1) / np.sqrt(S)))
                     for v in VFS} for g in GOALS}
        sl[m]["avg_goals"] = {v: dict(mean=float(np.mean([np.mean(ps[g][v]) for g in GOALS])),
                                      ci=boot_mean_ci(np.mean([ps[g][v] for g in GOALS], 0))) for v in VFS}
        sl[m]["avg_9"] = dict(mean=float(ps_mat(cap).mean()), ci=boot_mean_ci(ps_mat(cap)))
    out["slateN"] = sl
    ep, sl_ = rows["rms_persistence"], rows["slate"]
    A = {m: dict(accuracy=acc(e, ep), slate_ci=boot_acc(e, ep, sl_)) for m, e in err.items()}
    A["lf_v2_switched"]["harness"] = M["lf_v2_switched"]["accuracy"]
    A["lf_v2_single"]["harness"] = M["lf_v2_single"]["accuracy"]
    A["persistence"] = dict(accuracy=0.0)
    for a, b in [("lf_v2_switched", "lf_v1_switched"), ("lf_v2_single", "lf_v1_single"),
                 ("lf_v2_switched", "lf_v2_single"), ("lf_v2_switched", "nfd_v2_s0"),
                 ("lf_v2_switched", "lf_v2_switched_collection_bins")]:
        A[f"delta {a} - {b}"] = dict(delta=acc(err[a], ep) - acc(err[b], ep),
                                     slate_ci=boot_acc(err[a], ep, sl_, err[b]))
    out["accuracy_ds0019"] = A
    vr = dict(np.load(R / "lf_ds0020v2_val_rows.npz"))
    vn = dict(np.load(R / "ds0020v2_val_rows.npz"))
    vep = vr["rms_persistence"]
    V = {m: dict(accuracy=acc(vr[f"rms_{m}"], vep), traj_ci=boot_acc(vr[f"rms_{m}"], vep, vr["traj"]))
         for m in MODELS}
    V["nfd_v2_s0"] = dict(accuracy=acc(vn["rms_nfd_v2_s0"], vep), traj_ci=boot_acc(vn["rms_nfd_v2_s0"], vep, vr["traj"]))
    V["delta equal - collection bins (val, chosen-on-val)"] = dict(
        delta=acc(vr["rms_lf_v2_switched"], vep) - acc(vr["rms_lf_v2_switched_collection_bins"], vep),
        traj_ci=boot_acc(vr["rms_lf_v2_switched"], vep, vr["traj"], vr["rms_lf_v2_switched_collection_bins"]))
    V["n_rows"], V["persistence"] = int(len(vep)), 0.0
    out["accuracy_ds0020v2_val"] = V
    # accuracy per DS-0019 push-length bin (chosen scheme)
    import torch
    from Baselines.LinearForesight.model import bin_index
    ed = torch.tensor(raw["bin_counts_ds0019"]["equal"]["edges"])
    bb = bin_index(torch.tensor(rows["push_length"]), ed).numpy()
    out["accuracy_ds0019_per_bin"] = {int(b): dict(n=int((bb == b).sum()), **{m: acc(e[bb == b], ep[bb == b])
                                                    for m, e in err.items()}) for b in range(len(ed) - 1)}
    # paired slateN
    pnames = ["lf_v2_switched", "lf_v2_single", "lf_v1_switched", "lf_v1_single", "nfd_v2_s0"]
    pcaps = [caps["lf_v2_switched"], caps["lf_v2_single"], caps["lf_v1_switched (MODEL-0004, EXP-0061 file)"],
             caps["lf_v1_single (MODEL-0004, EXP-0061 file)"], caps["nfd_v2_s0 (MODEL-0008, RUN-0001 file)"]]
    keep = {("lf_v2_switched", "lf_v1_switched"), ("lf_v2_single", "lf_v1_single"),
            ("lf_v2_switched", "lf_v2_single"), ("lf_v2_switched", "nfd_v2_s0"), ("lf_v2_single", "nfd_v2_s0")}
    out["paired_slateN"] = {}
    for lab, vfs in [("all_9_cells", VFS)] + [(v, (v,)) for v in VFS]:
        X = np.stack([ps_mat(c, GOALS, vfs) for c in pcaps])
        out["paired_slateN"][lab] = [p for p in paired_comparison(X, pnames, n_boot=10000, seed=0)
                                     if (p["a"], p["b"]) in keep]
    out["paired_slateN_per_goal"] = {
        g: [p for p in paired_comparison(np.stack([ps_mat(c, (g,)) for c in pcaps]), pnames, n_boot=10000, seed=0)
            if (p["a"], p["b"]) in keep] for g in GOALS}
    out["paired_slateN_collection_vs_equal"] = paired_comparison(
        np.stack([ps_mat(caps["lf_v2_switched"]), ps_mat(caps["lf_v2_switched_collection_bins"])]),
        ["lf_v2_switched(equal)", "lf_v2_switched(collection)"], n_boot=10000, seed=0)[0]
    # strata
    meta = {int(k): v for k, v in d61["slate_meta"].items()}
    nrig_s = np.array([meta[int(s)]["n_rigids"] for s in sids])
    t1, t2 = np.quantile(nrig_s, [1 / 3, 2 / 3])
    terc_s = np.digitize(nrig_s, [t1, t2])
    blob_s = np.array([meta[int(s)]["init_pos"] == "rand_blob" for s in sids])
    blob_r, terc_r = r61["init_pos_blob"], np.digitize(r61["n_rigids"], [t1, t2])
    defs = {"rand_blob": (blob_r, blob_s), "rand_spread": (~blob_r, ~blob_s)}
    for k, lab in enumerate(["small", "mid", "large"]):
        defs[f"pieces_{lab}"] = (terc_r == k, terc_s == k)
    P = {n: ps_mat(c) for n, c in zip(pnames, pcaps)}
    st = dict(tercile_edges_n_rigids=[float(t1), float(t2)], strata={})
    for name, (rm, sm) in defs.items():
        s = dict(n_slates=int(sm.sum()), n_rows=int(rm.sum()))
        for n in pnames:
            s[f"slateN_{n}"] = float(P[n][sm].mean())
            s[f"acc_{n}"] = acc(err[n][rm], ep[rm])
        for a, b in [("lf_v2_switched", "lf_v1_switched"), ("lf_v2_switched", "lf_v2_single"),
                     ("lf_v2_switched", "nfd_v2_s0")]:
            d = P[a][sm] - P[b][sm]
            s[f"slateN_delta {a} - {b}"] = dict(mean=float(d.mean()), ci=boot_mean_ci(d))
        st["strata"][name] = s
    out["strata"] = st
    out["degeneracy_frac_dv_true_zero"] = {g: {v: d61["degeneracy"][g][v]["frac_dv_true_zero"] for v in VFS} for g in GOALS}
    _save("lf_ds0019", out)
    write_md(out)


def write_md(o):
    L = ["# EXP-0062 RUN-0002 -- linear visual foresight fit on DS-0020 v2 (MODEL-0009) on DS-0019", "",
         f"Truth: {o['truth']}. Truth check: values {o['truth_check']['unique_values']}.",
         f"{o['n_slates']} slates, {o['n_rows']} rows. v1 = MODEL-0004 (EXP-0061 per-slate file; re-scored here: "
         + "; ".join(f"{k}: max |per-slate diff| {v['max_abs_per_slate_diff']:.2g}, max |row rms diff| "
                     f"{v['max_abs_row_rms_diff']:.2g}, acc {v['accuracy_old']:.4f} -> {v['accuracy_new']:.4f}"
                     for k, v in o["reuse_check_MODEL_0004"].items()) + ").", "",
         "## DS-0019 rows per push-length bin", "", "| scheme | edges | rows/bin |", "|---|---|---|"]
    for k, v in o["bin_counts_ds0019"].items():
        L.append(f"| {k} | {[round(e, 3) for e in v['edges']]} | {v['counts']} |")
    L += ["", "## slateN, mean of 3 goals [slate-bootstrap 95% CI]", "",
          "| model | lyapunov | mass_in_region | signed_mass | all 9 [CI] |", "|---|---|---|---|---|"]
    for m, s in o["slateN"].items():
        c = [f"{s['avg_goals'][v]['mean']:.3f} [{s['avg_goals'][v]['ci'][0]:.3f}, {s['avg_goals'][v]['ci'][1]:.3f}]" for v in VFS]
        L.append(f"| {m} | " + " | ".join(c) + f" | {s['avg_9']['mean']:.3f} [{s['avg_9']['ci'][0]:.3f}, {s['avg_9']['ci'][1]:.3f}] |")
    L += ["", "## slateN per goal x vf (mean +- slate sem)", "",
          "| model | " + " | ".join(f"{g[:6]}/{v[:6]}" for g in GOALS for v in VFS) + " |", "|---" * 10 + "|"]
    for m, s in o["slateN"].items():
        L.append(f"| {m} | " + " | ".join(f"{s[g][v]['mean']:.3f}+-{s[g][v]['sem']:.3f}" for g in GOALS for v in VFS) + " |")
    a, va = o["accuracy_ds0019"], o["accuracy_ds0020v2_val"]
    L += ["", "## accuracy (swept region, ratio of population means) -- SUSPECT across model types", "",
          "| model | DS-0019 | slate CI | DS-0020 v2 val | traj CI |", "|---|---|---|---|---|"]
    vmap = {"lf_v1_switched": "lf_v1_switched_rescored", "lf_v1_single": "lf_v1_single_rescored"}
    for m in ["lf_v2_switched", "lf_v2_single", "lf_v2_switched_collection_bins", "lf_v1_switched", "lf_v1_single", "nfd_v2_s0"]:
        v = va[vmap.get(m, m)]
        L.append(f"| {m} | {a[m]['accuracy']:.4f} | {a[m]['slate_ci'][0]:.3f}-{a[m]['slate_ci'][1]:.3f} | "
                 f"{v['accuracy']:.4f} | {v['traj_ci'][0]:.3f}-{v['traj_ci'][1]:.3f} |")
    L.append("| persistence | 0 | | 0 | |")
    L += [""] + [f"- {k}: {v['delta']:+.4f} [{v['slate_ci'][0]:+.4f}, {v['slate_ci'][1]:+.4f}] (DS-0019, slate-cluster)"
                 for k, v in a.items() if k.startswith("delta")]
    dv = va["delta equal - collection bins (val, chosen-on-val)"]
    L += [f"- val: equal - collection bins: {dv['delta']:+.4f} [{dv['traj_ci'][0]:+.4f}, {dv['traj_ci'][1]:+.4f}] (traj-cluster)",
          f"- DS-0020 v2 val rows: {va['n_rows']}", "",
          "## DS-0019 accuracy per push-length bin (chosen equal-width scheme)", "",
          "| bin | n | " + " | ".join(next(iter(o["accuracy_ds0019_per_bin"].values())).keys() - {"n"}) + " |"]
    keys = [k for k in next(iter(o["accuracy_ds0019_per_bin"].values())) if k != "n"]
    L[-1] = "| bin | n | " + " | ".join(keys) + " |"
    L.append("|---" * (len(keys) + 2) + "|")
    for b, r in o["accuracy_ds0019_per_bin"].items():
        L.append(f"| {b} | {r['n']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    L += ["", "## paired slateN (paired_stats.paired_comparison; per-slate mean of the cells; Holm over the 10 pairs of "
          "5 models per row group, 5 shown)", "", "| cells | a - b | diff [95% CI] | wins/losses | Holm p |", "|---|---|---|---|---|"]
    for lab, prs in o["paired_slateN"].items():
        for p in prs:
            L.append(f"| {lab} | {p['a']} - {p['b']} | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    for g, prs in o["paired_slateN_per_goal"].items():
        for p in prs:
            L.append(f"| goal {g} | {p['a']} - {p['b']} | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    p = o["paired_slateN_collection_vs_equal"]
    L.append(f"| all 9 | {p['a']} - {p['b']} | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
             f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    L += ["", f"## strata (piece-count tercile edges {o['strata']['tercile_edges_n_rigids']}; slateN = per-slate mean of 9 cells)", "",
          "| stratum (slates) | slateN LF v2 sw / single / LF v1 sw / NFD v2 | d(v2sw - v1sw) | d(v2sw - v2single) | d(v2sw - NFDv2) | acc LF v2 sw / v1 sw / NFD v2 |",
          "|---|---|---|---|---|---|"]
    for n, s in o["strata"]["strata"].items():
        ds = [s[f"slateN_delta {a_} - {b_}"] for a_, b_ in [("lf_v2_switched", "lf_v1_switched"),
                                                             ("lf_v2_switched", "lf_v2_single"), ("lf_v2_switched", "nfd_v2_s0")]]
        L.append(f"| {n} ({s['n_slates']}) | {s['slateN_lf_v2_switched']:.3f} / {s['slateN_lf_v2_single']:.3f} / "
                 f"{s['slateN_lf_v1_switched']:.3f} / {s['slateN_nfd_v2_s0']:.3f} | "
                 + " | ".join(f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" for d in ds)
                 + f" | {s['acc_lf_v2_switched']:.3f} / {s['acc_lf_v1_switched']:.3f} / {s['acc_nfd_v2_s0']:.3f} |")
    tmp = R / "lf_ds0019.md.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, R / "lf_ds0019.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["score", "analyze"])
    a = ap.parse_args()
    score(a) if a.what == "score" else analyze(a)
