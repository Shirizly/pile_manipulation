"""EXP-0062 -- score the DS-0020-v2-trained NFD on DS-0019 (and DS-0020 v2 val) and
compare it, paired over the same 100 slates, with EXP-0061's v1-trained NFD (MODEL-0005).

Same code path and flags as EXP-0061 RUN-0001 (`code/final_eval.py` there): corpus
`flex_ds0019_mask` (occ0 = binary image mask), truth = cell.occ1 = the BINARY image
mask of the true after-state (= eval_report `--truth-scoring image`, i.e.
`_capture_report(cell, pred, "default", None)`), goals random_quadrant/ring_O/T x
lyapunov/mass_in_region/signed_mass, swept-region accuracy via the per-row rms
(ratio of population means). EXP-0061's per-slate file (results/final_eval/ds0019.json
+ ds0019_rows.npz) is REUSED for MODEL-0005, persistence, random, slate metadata and
degeneracy; MODEL-0005 is also re-scored here once as a reproduction check.

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_nfd_v2.py score \
        --ckpt Baselines/NFD/runs/nfd_3ch_flex_mask_v2_seed0/unet_best.pth --name nfd_v2_s0
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_nfd_v2.py analyze --name nfd_v2_s0

Outputs (atomic): results/nfd_seed0_ds0019_raw.json + _rows.npz (score), results/
nfd_seed0_ds0019.{json,md} (analyze).
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
V1_CKPT = "Baselines/NFD/runs/nfd_3ch_flex_mask_seed0/unet_best.pth"     # MODEL-0005
D20V2 = "datasets/DS-0020-training-data-flex-N864/config.yaml"
GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
NB = 2000


def _save(tag, res, rows=None):
    from Baselines.common import eval_report as er
    R.mkdir(parents=True, exist_ok=True)
    er._write_json_atomic(str(R / f"{tag}.json"), res)
    if rows is not None:
        tmp = R / f"{tag}_rows.tmp.npz"
        np.savez(tmp, **rows)
        os.replace(tmp, R / f"{tag}_rows.npz")


def score(args):
    import torch
    from Baselines.common import eval_report as er
    from FlexData.dataset import load_flex_cell
    from fit_linear_foresight import metrics
    sys.path.insert(0, str(E61 / "code"))
    from final_eval import region_of, row_rms
    from score_nfd_flex import nfd_spec

    res = dict(truth="binary image mask of the true after-state (cell.occ1, occ_source=image_mask); "
                     "eval_report._capture_report(truth_s0=None) == --truth-scoring image",
               models={}, val={})
    # ---- DS-0019
    t0 = time.time()
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    o1 = cell.occ1
    assert o1.dtype in (torch.float32, torch.uint8) and set(torch.unique(o1).tolist()) <= {0.0, 1.0}, \
        "truth is not a binary mask"
    res["truth_check"] = dict(unique_values=sorted(torch.unique(o1).tolist()), dtype=str(o1.dtype),
                              occ_source=cell.raw.occ_source if hasattr(cell.raw, "occ_source") else "image_mask")
    region = region_of(cell)
    truth, prev = cell.occ1.float(), cell.occ0.float()
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    rows = dict(slate=cell.slate_idx.numpy(), rms_persistence=row_rms(prev, truth, region).numpy())
    assert np.array_equal(rows["slate"], r61["slate"]), "row order differs from EXP-0061"
    assert np.allclose(rows["rms_persistence"], r61["rms_persistence"]), "persistence errors differ from EXP-0061"
    print(f"[ds0019] {len(cell.occ0)} rows, row order + persistence == EXP-0061 ({time.time() - t0:.0f}s)", flush=True)
    for name, ck in [(args.name, args.ckpt), ("nfd_v1_s0_rescored", V1_CKPT)]:
        t1 = time.time()
        spec = nfd_spec(ck)
        acc, pred = er._accuracy(spec, er._load_predictor(spec), cell, "cuda")
        rows[f"rms_{name}"] = row_rms(pred.float(), truth, region).numpy()
        cap = er._capture_report(cell, pred, "default", None)
        res["models"][name] = dict(ckpt=ck, accuracy=acc, capture=cap, seconds=time.time() - t1)
        print(name, f"acc {acc:.4f}", {k: round(v, 4) for k, v in cap["averaged_over_goals"].items()}, flush=True)
        _save("nfd_seed0_ds0019_raw", res, rows)
    # ---- DS-0020 v2 val (in-domain val accuracy; v1 model too, for reference)
    vcell = load_flex_cell(D20V2, "val", tag="ds0020v2_val_mask", occ_source="image_mask")
    vregion = region_of(vcell)
    vt, vp = vcell.occ1.float(), vcell.occ0.float()
    vrows = dict(traj=vcell.slate_idx.numpy(), rms_persistence=row_rms(vp, vt, vregion).numpy())
    res["val"]["n_rows"] = int(len(vcell.occ0))
    res["val"]["persistence_accuracy"] = metrics(vp, vt, vp, region=vregion)["accuracy"]
    for name, ck in [(args.name, args.ckpt), ("nfd_v1_s0_rescored", V1_CKPT)]:
        spec = nfd_spec(ck)
        acc, pred = er._accuracy(spec, er._load_predictor(spec), vcell, "cuda")
        vrows[f"rms_{name}"] = row_rms(pred.float(), vt, vregion).numpy()
        res["val"][name] = dict(accuracy=acc)
        print("val", name, f"acc {acc:.4f}", flush=True)
    _save("nfd_seed0_ds0019_raw", res, rows)
    tmp = R / "ds0020v2_val_rows.tmp.npz"
    np.savez(tmp, **vrows)
    os.replace(tmp, R / "ds0020v2_val_rows.npz")


# ------------------------------------------------------------------ analysis
rng = np.random.default_rng(0)


def acc(e, ep):
    return float(1 - e.mean() / ep.mean())


def boot_acc(e, ep, groups, e2=None):
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    sm, sp, cnt = np.bincount(inv, e, G), np.bincount(inv, ep, G), np.bincount(inv, None, G)
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
    raw = json.load(open(R / "nfd_seed0_ds0019_raw.json"))
    rows = dict(np.load(R / "nfd_seed0_ds0019_raw_rows.npz"))
    d61 = json.load(open(R61 / "ds0019.json"))
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    v2, v1r = args.name, "nfd_v1_s0_rescored"
    caps = {v2: raw["models"][v2]["capture"], "nfd_v1_s0 (EXP-0061 file)": d61["models"]["nfd_s0"]["capture"],
            "random": d61["models"]["random"]["capture"], "persistence": d61["models"]["persistence"]["capture"]}
    sids = caps[v2]["slate_ids"]
    assert sids == d61["models"]["nfd_s0"]["capture"]["slate_ids"]
    # reproduction check of the reused EXP-0061 MODEL-0005 numbers
    c_old, c_new = d61["models"]["nfd_s0"]["capture"], raw["models"][v1r]["capture"]
    repro = dict(accuracy_old=d61["models"]["nfd_s0"]["accuracy"], accuracy_new=raw["models"][v1r]["accuracy"],
                 max_abs_per_slate_diff=float(max(np.abs(np.array(c_old["per_slate"][g][v]) -
                                                         np.array(c_new["per_slate"][g][v])).max()
                                                  for g in GOALS for v in VFS)),
                 max_abs_row_rms_diff=float(np.abs(r61["rms_nfd_s0"] - rows[f"rms_{v1r}"]).max()))
    S = len(sids)
    out = dict(model=v2, ckpt=raw["models"][v2]["ckpt"], n_slates=S, n_rows=int(len(rows["slate"])),
               truth=raw["truth"], truth_check=raw["truth_check"], reuse_check_MODEL_0005=repro)
    # slateN tables
    sl = {}
    for m, cap in caps.items():
        ps = cap["per_slate"]
        sl[m] = {g: {v: dict(mean=float(np.mean(ps[g][v])), sem=float(np.std(ps[g][v], ddof=1) / np.sqrt(S)))
                     for v in VFS} for g in GOALS}
        sl[m]["avg_goals"] = {v: dict(mean=float(np.mean([np.mean(ps[g][v]) for g in GOALS])),
                                      ci=boot_mean_ci(np.mean([ps[g][v] for g in GOALS], 0))) for v in VFS}
        sl[m]["avg_9"] = dict(mean=float(ps_mat(cap).mean()), ci=boot_mean_ci(ps_mat(cap)))
    out["slateN"] = sl
    # accuracy
    ep = rows["rms_persistence"]
    e2, e1 = rows[f"rms_{v2}"], r61["rms_nfd_s0"]
    out["accuracy_ds0019"] = {
        v2: dict(accuracy=acc(e2, ep), harness=raw["models"][v2]["accuracy"], slate_ci=boot_acc(e2, ep, rows["slate"])),
        "nfd_v1_s0": dict(accuracy=acc(e1, ep), slate_ci=boot_acc(e1, ep, rows["slate"])),
        "persistence": dict(accuracy=0.0),
        "delta_v2_minus_v1": dict(delta=acc(e2, ep) - acc(e1, ep), slate_ci=boot_acc(e2, ep, rows["slate"], e1))}
    # val
    vr = dict(np.load(R / "ds0020v2_val_rows.npz"))
    vep = vr["rms_persistence"]
    out["accuracy_ds0020v2_val"] = {m: dict(accuracy=acc(vr[f"rms_{m}"], vep), traj_ci=boot_acc(vr[f"rms_{m}"], vep, vr["traj"]))
                                    for m in (v2, v1r)}
    out["accuracy_ds0020v2_val"]["n_rows"] = int(len(vep))
    out["accuracy_ds0020v2_val"]["persistence"] = 0.0
    # paired slateN v2 - v1 (+ each vs random)
    X = {lab: np.stack([ps_mat(caps[v2], GOALS, vfs), ps_mat(caps["nfd_v1_s0 (EXP-0061 file)"], GOALS, vfs),
                        ps_mat(caps["random"], GOALS, vfs)])
         for lab, vfs in [("all_9_cells", VFS)] + [(v, (v,)) for v in VFS]}
    out["paired_slateN"] = {lab: paired_comparison(x, [v2, "nfd_v1_s0", "random"], n_boot=10000, seed=0)
                            for lab, x in X.items()}
    out["paired_slateN_per_goal"] = {
        g: paired_comparison(np.stack([ps_mat(caps[v2], (g,)), ps_mat(caps["nfd_v1_s0 (EXP-0061 file)"], (g,))]),
                             [v2, "nfd_v1_s0"], n_boot=10000, seed=0)[0] for g in GOALS}
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
    p2, p1 = ps_mat(caps[v2]), ps_mat(caps["nfd_v1_s0 (EXP-0061 file)"])
    st = dict(tercile_edges_n_rigids=[float(t1), float(t2)], strata={})
    for name, (rm, sm) in defs.items():
        d = p2[sm] - p1[sm]
        st["strata"][name] = dict(
            n_slates=int(sm.sum()), n_rows=int(rm.sum()),
            slateN_v2=float(p2[sm].mean()), slateN_v1=float(p1[sm].mean()),
            slateN_delta=float(d.mean()), slateN_delta_ci=boot_mean_ci(d),
            acc_v2=acc(e2[rm], ep[rm]), acc_v1=acc(e1[rm], ep[rm]),
            acc_delta_slate_ci=boot_acc(e2[rm], ep[rm], rows["slate"][rm], e1[rm]))
    out["strata"] = st
    out["degeneracy_frac_dv_true_zero"] = {g: {v: d61["degeneracy"][g][v]["frac_dv_true_zero"] for v in VFS} for g in GOALS}
    out["degeneracy_frac_slates_flat"] = {g: {v: d61["degeneracy"][g][v]["frac_slates_flat"] for v in VFS} for g in GOALS}
    _save("nfd_seed0_ds0019", out)
    write_md(out, v2)


def write_md(o, v2):
    L = [f"# EXP-0062 -- {v2} (DS-0020 v2-trained NFD, seed 0) on DS-0019", "",
         f"Truth: {o['truth']}. Truth check: values {o['truth_check']['unique_values']}.",
         f"{o['n_slates']} slates, {o['n_rows']} rows. v1 = MODEL-0005 (EXP-0061 per-slate file, reproduced: "
         f"max |per-slate diff| {o['reuse_check_MODEL_0005']['max_abs_per_slate_diff']:.2g}, "
         f"accuracy {o['reuse_check_MODEL_0005']['accuracy_old']:.4f} -> {o['reuse_check_MODEL_0005']['accuracy_new']:.4f}).", "",
         "## slateN, mean of 3 goals [slate-bootstrap 95% CI]", "",
         "| model | lyapunov | mass_in_region | signed_mass | all 9 |", "|---|---|---|---|---|"]
    for m, s in o["slateN"].items():
        c = [f"{s['avg_goals'][v]['mean']:.3f} [{s['avg_goals'][v]['ci'][0]:.3f}, {s['avg_goals'][v]['ci'][1]:.3f}]" for v in VFS]
        L.append(f"| {m} | " + " | ".join(c) + f" | {s['avg_9']['mean']:.3f} |")
    L += ["", "## slateN per goal x vf (mean +- slate sem)", "",
          "| model | " + " | ".join(f"{g[:6]}/{v[:6]}" for g in GOALS for v in VFS) + " |", "|---" * 10 + "|"]
    for m, s in o["slateN"].items():
        L.append(f"| {m} | " + " | ".join(f"{s[g][v]['mean']:.3f}+-{s[g][v]['sem']:.3f}" for g in GOALS for v in VFS) + " |")
    a = o["accuracy_ds0019"]
    L += ["", "## accuracy (swept region, ratio of population means; slate-cluster bootstrap CI)", "",
          "| model | DS-0019 | slate CI | DS-0020 v2 val | traj CI |", "|---|---|---|---|---|"]
    va = o["accuracy_ds0020v2_val"]
    L.append(f"| {v2} | {a[v2]['accuracy']:.4f} | {a[v2]['slate_ci'][0]:.3f}-{a[v2]['slate_ci'][1]:.3f} | "
             f"{va[v2]['accuracy']:.4f} | {va[v2]['traj_ci'][0]:.3f}-{va[v2]['traj_ci'][1]:.3f} |")
    L.append(f"| nfd_v1_s0 (MODEL-0005) | {a['nfd_v1_s0']['accuracy']:.4f} | {a['nfd_v1_s0']['slate_ci'][0]:.3f}-"
             f"{a['nfd_v1_s0']['slate_ci'][1]:.3f} | {va['nfd_v1_s0_rescored']['accuracy']:.4f} | "
             f"{va['nfd_v1_s0_rescored']['traj_ci'][0]:.3f}-{va['nfd_v1_s0_rescored']['traj_ci'][1]:.3f} |")
    L.append("| persistence | 0 | | 0 | |")
    d = a["delta_v2_minus_v1"]
    L += ["", f"Paired accuracy delta v2 - v1 on DS-0019: {d['delta']:+.4f} [{d['slate_ci'][0]:+.4f}, {d['slate_ci'][1]:+.4f}] (slate-cluster).",
          f"DS-0020 v2 val rows: {va['n_rows']}.", "",
          "## paired slateN v2 - v1 (paired_stats.paired_comparison; per-slate mean of the cells; 100 slates)", "",
          "| cells | diff [95% CI] | wins/losses | Holm p |", "|---|---|---|---|"]
    for lab, prs in o["paired_slateN"].items():
        for p in prs:
            L.append(f"| {lab}: {p['a']} - {p['b']} | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    for g, p in o["paired_slateN_per_goal"].items():
        L.append(f"| goal {g} (3 vf): v2 - v1 | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                 f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    L += ["", f"## strata (piece-count tercile edges {o['strata']['tercile_edges_n_rigids']})", "",
          "| stratum (slates) | slateN v2 | slateN v1 | delta [CI] | acc v2 | acc v1 | acc delta CI |", "|---|---|---|---|---|---|---|"]
    for n, s in o["strata"]["strata"].items():
        L.append(f"| {n} ({s['n_slates']}) | {s['slateN_v2']:.3f} | {s['slateN_v1']:.3f} | {s['slateN_delta']:+.3f} "
                 f"[{s['slateN_delta_ci'][0]:+.3f}, {s['slateN_delta_ci'][1]:+.3f}] | {s['acc_v2']:.3f} | {s['acc_v1']:.3f} | "
                 f"[{s['acc_delta_slate_ci'][0]:+.3f}, {s['acc_delta_slate_ci'][1]:+.3f}] |")
    L += ["", "## goal degeneracy frac(dv_true == 0) (binary-mask truth; from EXP-0061, same rows)", "",
          "| goal | " + " | ".join(VFS) + " |", "|---|---|---|---|"]
    for g in GOALS:
        L.append(f"| {g} | " + " | ".join(f"{o['degeneracy_frac_dv_true_zero'][g][v]:.4f}" for v in VFS) + " |")
    tmp = R / "nfd_seed0_ds0019.md.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, R / "nfd_seed0_ds0019.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["score", "analyze"])
    ap.add_argument("--ckpt", default="Baselines/NFD/runs/nfd_3ch_flex_mask_v2_seed0/unet_best.pth")
    ap.add_argument("--name", default="nfd_v2_s0")
    a = ap.parse_args()
    score(a) if a.what == "score" else analyze(a)
