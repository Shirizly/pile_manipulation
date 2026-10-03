"""EXP-0062 RUN-0003 -- score the dyn-res GNN trained from scratch on DS-0020 v2 at a constant
30 nodes (MODEL-0010, `Baselines/GNN/flex_train.py`) on DS-0019 and DS-0020 v2 val; paired slateN
vs v2 NFD (MODEL-0008, RUN-0001 file), v2 LF (MODEL-0009, RUN-0002 file), EXP-0061's original
checkpoint at N = 200 (EXP-0061 per-slate file, reused after a bit-identical re-score) and the
original checkpoint at N = 30 (scored here); + the nearest-node RENDERING CAP at N = 30 (every fg
pixel carried by its node's TRUE motion = the displacement of the node's nearest simulator
particle, table XY) and the small-pile fallback count.

Same code path / flags as RUN-0001/0002 and EXP-0061 final_eval.py: corpus flex_ds0019_mask,
truth = cell.occ1 = binary image mask of the true after-state (== eval_report --truth-scoring image,
`_capture_report(cell, pred, "default", None)`), goals random_quadrant/ring_O/T x lyapunov/
mass_in_region/signed_mass, swept-region accuracy from per-row rms (ratio of population means).

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_gnn_v2.py score --alt-ckpt PATH
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/eval_gnn_v2.py analyze
Outputs (atomic): results/gnn_ds0019_raw.json + _raw_rows.npz, results/gnn_ds0020v2_val_rows.npz,
results/gnn_ds0019.{json,md}.
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
D20V2 = "datasets/DS-0020-training-data-flex-N864/config.yaml"
GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
NB = 2000
import torch as _t
DEV = "cuda" if _t.cuda.is_available() else "cpu"   # the GPU failed ("requires reset") during RUN-0003 -> cpu


def _save_json(name, obj):
    from Baselines.common import eval_report as er
    R.mkdir(parents=True, exist_ok=True)
    er._write_json_atomic(str(R / name), obj)


def _save_npz(name, rows):
    tmp = R / (name + ".tmp.npz")
    np.savez(tmp, **rows)
    os.replace(tmp, R / name)


def models(alt_ckpt):
    from Baselines.common import eval_report as er
    m = {"gnn_v2_n30": dict(er.MODELS["gnn_flex_v2_n30"]),
         "gnn_old_n200_rescored": dict(er.MODELS["gnn_flex_drp"]),
         "gnn_old_n30": dict(er.MODELS["gnn_flex_drp_n30"])}
    if alt_ckpt:
        m["gnn_v2_n30_alt_target"] = dict(er.MODELS["gnn_flex_v2_n30"], ckpt=alt_ckpt)
    return m


def true_motion_cap(cell, n=30, device="cuda", ds_index=None):
    """Render with every node's TRUE motion: nodes from the predictor's own perception (N = n),
    displacement = after - before of each node's nearest particle (table XY). Returns (pred,
    n_fallback_states, n_states)."""
    import torch
    from scipy.spatial import cKDTree
    from Baselines.GNN.flex_predictor import FlexGNNPredictor, GLOBAL_SCALE
    p = FlexGNNPredictor(particle_num=n, depth_mode="plane", plane_y=0.24, carry_k=1, device=device)
    raw = cell.raw
    H, W = cell.H, cell.W
    to_pxl, ctr = float(raw.to_pxl), float(raw.ctr_in_PXL[0])
    out = torch.zeros(len(cell.occ0), H, W)
    groups = {}
    for i, (g, s) in enumerate(zip(cell.run_idx.tolist(), cell.step_idx.tolist())):
        groups.setdefault((g, s), []).append(i)
    nfb = 0
    for (g, s), rows in groups.items():
        st = p._state(raw, int(g), int(s))
        nfb += bool(st["fallback"])
        nodes = torch.from_numpy(st["nodes"]).to(device)
        q = np.stack([st["nodes"][:, 0], -st["nodes"][:, 1]], 1) * GLOBAL_SCALE      # table XY
        ri = (lambda r: r) if ds_index is None else (lambda r: int(ds_index[r]))   # cell row -> dataset idx
        p0 = raw.particles_before(ri(rows[0])).numpy()
        ok = np.isfinite(p0).all(1)
        idx = np.nonzero(ok)[0][cKDTree(p0[ok]).query(q)[1]]
        disp = np.stack([(raw.particles_after(ri(r)).numpy()[idx] - p0[idx]) for r in rows])   # (B,N,2) table
        disp = np.nan_to_num(disp)
        out[rows] = p.render(st, torch.from_numpy(disp).float().to(device), nodes, H, W, to_pxl, ctr).cpu()
    return out, nfb, len(groups)


def score(args):
    import torch
    from Baselines.common import eval_report as er
    from FlexData.dataset import load_flex_cell
    from fit_linear_foresight import metrics
    sys.path.insert(0, str(E61 / "code"))
    from final_eval import region_of, row_rms

    M = models(args.alt_ckpt)
    res = dict(truth="binary image mask of the true after-state (cell.occ1, occ_source=image_mask); "
                     "eval_report._capture_report(truth_s0=None) == --truth-scoring image",
               models={}, val={}, ckpts={k: v["ckpt"] for k, v in M.items()},
               kwargs={k: v["kwargs"] for k, v in M.items()})
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    o1 = cell.occ1
    assert set(torch.unique(o1).tolist()) <= {0.0, 1.0}, "truth is not a binary mask"
    res["truth_check"] = dict(unique_values=sorted(torch.unique(o1).tolist()))
    region = region_of(cell)
    truth, prev = cell.occ1.float(), cell.occ0.float()
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    rows = dict(slate=cell.slate_idx.numpy(), push_length=cell.push_length.numpy(),
                rms_persistence=row_rms(prev, truth, region).numpy())
    assert np.array_equal(rows["slate"], r61["slate"]) and np.allclose(rows["rms_persistence"], r61["rms_persistence"])
    done, vdone = set(), set()
    if (R / "gnn_ds0019_raw.json").exists() and args.resume:     # keep finished models (same ckpt)
        old = json.load(open(R / "gnn_ds0019_raw.json"))
        oldrows = dict(np.load(R / "gnn_ds0019_raw_rows.npz"))
        for k, v in old["models"].items():
            if k in M and v["ckpt"] == M[k]["ckpt"]:
                res["models"][k] = v; rows[f"rms_{k}"] = oldrows[f"rms_{k}"]; done.add(k)
        if old.get("cap"):
            res["cap"] = old["cap"]; rows["rms_cap_n30"] = oldrows["rms_cap_n30"]
        if (R / "gnn_ds0020v2_val_rows.npz").exists():
            ovr = dict(np.load(R / "gnn_ds0020v2_val_rows.npz"))
            for k, v in old.get("val", {}).items():
                if isinstance(v, dict) and (k == "cap_n30" or (k in M and v.get("ckpt") == M[k]["ckpt"])):
                    res["val"][k] = v; vdone.add(k); res["val"].setdefault("_rows", {})[k] = ovr[f"rms_{k}"].tolist()
    only = set(args.only.split(",")) if args.only else None
    for name, spec in M.items():
        if name in done or (only and name not in only):
            continue
        t1 = time.time()
        pr = er._load_predictor(spec)
        acc, pred = er._accuracy(spec, pr, cell, DEV)
        rows[f"rms_{name}"] = row_rms(pred.float(), truth, region).numpy()
        cap = er._capture_report(cell, pred, "default", None)
        nfb = sum(bool(v["fallback"]) for v in pr._cache.values())
        res["models"][name] = dict(ckpt=spec["ckpt"], kwargs=spec["kwargs"], accuracy=acc, capture=cap,
                                   fallback_states=nfb, states_perceived=len(pr._cache), seconds=time.time() - t1)
        print(name, f"acc {acc:.4f} fallback {nfb}/{len(pr._cache)}",
              {k: round(v, 4) for k, v in cap["averaged_over_goals"].items()}, f"{time.time() - t1:.0f}s", flush=True)
        _save_json("gnn_ds0019_raw.json", res); _save_npz("gnn_ds0019_raw_rows.npz", rows)
        del pred
    if not res.get("cap"):
        t1 = time.time()
        capred, nfb, ns = true_motion_cap(cell, 30, DEV)
        rows["rms_cap_n30"] = row_rms(capred.float(), truth, region).numpy()
        res["cap"] = dict(n=30, accuracy=metrics(capred, truth, prev, region=region)["accuracy"],
                          fallback_states=nfb, n_states=ns,
                          capture=er._capture_report(cell, capred, "default", None))
        print("cap_n30", res["cap"]["accuracy"], f"fallback {nfb}/{ns}", f"{time.time() - t1:.0f}s", flush=True)
        _save_json("gnn_ds0019_raw.json", res); _save_npz("gnn_ds0019_raw_rows.npz", rows)
    # DS-0020 v2 val
    vcell = load_flex_cell(D20V2, "val", tag="ds0020v2_val_mask", occ_source="image_mask")
    vregion = region_of(vcell)
    vt, vp = vcell.occ1.float(), vcell.occ0.float()
    vrows = dict(traj=vcell.slate_idx.numpy(), push_length=vcell.push_length.numpy(),
                 rms_persistence=row_rms(vp, vt, vregion).numpy())
    v62 = dict(np.load(R / "ds0020v2_val_rows.npz"))
    assert np.array_equal(vrows["traj"], v62["traj"]) and np.allclose(vrows["rms_persistence"], v62["rms_persistence"])
    res["val"]["n_rows"] = int(len(vcell.occ0))
    for k, r in res["val"].pop("_rows", {}).items():
        vrows[f"rms_{k}"] = np.asarray(r, np.float32)
    for name, spec in M.items():
        if name in vdone or (only and name not in only) or (args.skip_val_old and name.startswith("gnn_old_n200")):
            continue
        t1 = time.time()
        pr = er._load_predictor(spec)
        acc, pred = er._accuracy(spec, pr, vcell, DEV)
        vrows[f"rms_{name}"] = row_rms(pred.float(), vt, vregion).numpy()
        nfb = sum(bool(v["fallback"]) for v in pr._cache.values())
        res["val"][name] = dict(accuracy=acc, fallback_states=nfb, states=len(pr._cache), ckpt=spec["ckpt"])
        print("val", name, f"acc {acc:.4f} fallback {nfb}", f"{time.time() - t1:.0f}s", flush=True)
        _save_json("gnn_ds0019_raw.json", res); _save_npz("gnn_ds0020v2_val_rows.npz", vrows)
    if "cap_n30" not in vdone:
        capred, nfb, ns = true_motion_cap(vcell, 30, DEV)
        vrows["rms_cap_n30"] = row_rms(capred.float(), vt, vregion).numpy()
        res["val"]["cap_n30"] = dict(accuracy=metrics(capred, vt, vp, region=vregion)["accuracy"],
                                     fallback_states=nfb, n_states=ns)
        print("val cap_n30", res["val"]["cap_n30"], flush=True)
    _save_json("gnn_ds0019_raw.json", res); _save_npz("gnn_ds0020v2_val_rows.npz", vrows)


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
    raw = json.load(open(R / "gnn_ds0019_raw.json"))
    rows = dict(np.load(R / "gnn_ds0019_raw_rows.npz"))
    d61 = json.load(open(R61 / "ds0019.json"))
    r61 = dict(np.load(R61 / "ds0019_rows.npz"))
    nraw = json.load(open(R / "nfd_seed0_ds0019_raw.json")); nrows = dict(np.load(R / "nfd_seed0_ds0019_raw_rows.npz"))
    lraw = json.load(open(R / "lf_ds0019_raw.json")); lrows = dict(np.load(R / "lf_ds0019_raw_rows.npz"))
    assert np.array_equal(nrows["slate"], rows["slate"]) and np.array_equal(lrows["slate"], rows["slate"])
    M = raw["models"]
    # reuse check: old checkpoint N=200 re-scored here vs EXP-0061's gnn_flex_drp_samp0
    co, cn = d61["models"]["gnn_flex_drp_samp0"]["capture"], M["gnn_old_n200_rescored"]["capture"]
    repro = dict(accuracy_old=d61["models"]["gnn_flex_drp_samp0"]["accuracy"],
                 accuracy_new=M["gnn_old_n200_rescored"]["accuracy"],
                 max_abs_per_slate_diff=float(max(np.abs(np.array(co["per_slate"][g][v]) - np.array(cn["per_slate"][g][v])).max()
                                                  for g in GOALS for v in VFS)),
                 max_abs_row_rms_diff=float(np.abs(r61["rms_gnn_flex_drp_samp0"] - rows["rms_gnn_old_n200_rescored"]).max()))
    caps = {"gnn_v2_n30 (MODEL-0010)": M["gnn_v2_n30"]["capture"]}
    if "gnn_v2_n30_alt_target" in M:
        caps["gnn_v2_n30_alt_target"] = M["gnn_v2_n30_alt_target"]["capture"]
    caps.update({"gnn_old_n30 (orig ckpt, N 30)": M["gnn_old_n30"]["capture"],
                 "gnn_old_n200 (orig ckpt, EXP-0061 file)": co,
                 "cap_n30 (true node motion)": raw["cap"]["capture"],
                 "nfd_v2_s0 (MODEL-0008, RUN-0001 file)": nraw["models"]["nfd_v2_s0"]["capture"],
                 "lf_v2_switched (MODEL-0009, RUN-0002 file)": lraw["models"]["lf_v2_switched"]["capture"],
                 "random": d61["models"]["random"]["capture"], "persistence": d61["models"]["persistence"]["capture"]})
    sids = caps["gnn_v2_n30 (MODEL-0010)"]["slate_ids"]
    for c in caps.values():
        assert c["slate_ids"] == sids
    err = {"gnn_v2_n30": rows["rms_gnn_v2_n30"], "gnn_old_n30": rows["rms_gnn_old_n30"],
           "gnn_old_n200": r61["rms_gnn_flex_drp_samp0"], "cap_n30": rows["rms_cap_n30"],
           "nfd_v2_s0": nrows["rms_nfd_v2_s0"], "lf_v2_switched": lrows["rms_lf_v2_switched"]}
    if "rms_gnn_v2_n30_alt_target" in rows:
        err["gnn_v2_n30_alt_target"] = rows["rms_gnn_v2_n30_alt_target"]
    S = len(sids)
    out = dict(n_slates=S, n_rows=int(len(rows["slate"])), truth=raw["truth"], truth_check=raw["truth_check"],
               reuse_check_old_n200=repro, ckpts=raw["ckpts"],
               fallback={k: dict(states=v["fallback_states"], of=v["states_perceived"]) for k, v in M.items()},
               cap_fallback=dict(states=raw["cap"]["fallback_states"], of=raw["cap"]["n_states"]))
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
    A["persistence"] = dict(accuracy=0.0)
    pairs_acc = [("gnn_v2_n30", "gnn_old_n30"), ("gnn_v2_n30", "gnn_old_n200"), ("gnn_v2_n30", "nfd_v2_s0"),
                 ("gnn_v2_n30", "lf_v2_switched"), ("cap_n30", "gnn_v2_n30")]
    if "gnn_v2_n30_alt_target" in err:
        pairs_acc.append(("gnn_v2_n30", "gnn_v2_n30_alt_target"))
    for a, b in pairs_acc:
        A[f"delta {a} - {b}"] = dict(delta=acc(err[a], ep) - acc(err[b], ep), slate_ci=boot_acc(err[a], ep, sl_, err[b]))
    out["accuracy_ds0019"] = A
    out["dynamics_share_of_cap"] = A["gnn_v2_n30"]["accuracy"] / A["cap_n30"]["accuracy"]
    vr = dict(np.load(R / "gnn_ds0020v2_val_rows.npz"))
    vn = dict(np.load(R / "ds0020v2_val_rows.npz")); vl = dict(np.load(R / "lf_ds0020v2_val_rows.npz"))
    vep = vr["rms_persistence"]
    V = {k[4:]: dict(accuracy=acc(vr[k], vep), traj_ci=boot_acc(vr[k], vep, vr["traj"]))
         for k in vr if k.startswith("rms_") and k != "rms_persistence"}
    V["nfd_v2_s0"] = dict(accuracy=acc(vn["rms_nfd_v2_s0"], vep))
    V["lf_v2_switched"] = dict(accuracy=acc(vl["rms_lf_v2_switched"], vep))
    V["n_rows"], V["persistence"] = int(len(vep)), 0.0
    V["fallback"] = {k: v.get("fallback_states") for k, v in raw["val"].items() if isinstance(v, dict)}
    out["accuracy_ds0020v2_val"] = V
    # paired slateN
    pn = ["gnn_v2_n30", "gnn_old_n30", "gnn_old_n200", "nfd_v2_s0", "lf_v2_switched"]
    pc = [caps["gnn_v2_n30 (MODEL-0010)"], caps["gnn_old_n30 (orig ckpt, N 30)"],
          caps["gnn_old_n200 (orig ckpt, EXP-0061 file)"], caps["nfd_v2_s0 (MODEL-0008, RUN-0001 file)"],
          caps["lf_v2_switched (MODEL-0009, RUN-0002 file)"]]
    if "gnn_v2_n30_alt_target" in caps:
        pn.append("gnn_v2_n30_alt_target"); pc.append(caps["gnn_v2_n30_alt_target"])
    keep = {("gnn_v2_n30", b) for b in pn[1:]}
    out["paired_slateN"] = {}
    for lab, vfs in [("all_9_cells", VFS)] + [(v, (v,)) for v in VFS]:
        X = np.stack([ps_mat(c, GOALS, vfs) for c in pc])
        out["paired_slateN"][lab] = [p for p in paired_comparison(X, pn, n_boot=10000, seed=0) if (p["a"], p["b"]) in keep]
    out["paired_slateN_per_goal"] = {
        g: [p for p in paired_comparison(np.stack([ps_mat(c, (g,)) for c in pc]), pn, n_boot=10000, seed=0)
            if (p["a"], p["b"]) in keep] for g in GOALS}
    capv = ps_mat(raw["cap"]["capture"])
    out["slateN_cap_minus_gnn_v2"] = dict(mean=float((capv - ps_mat(pc[0])).mean()), ci=boot_mean_ci(capv - ps_mat(pc[0])))
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
    Pm = {n: ps_mat(c) for n, c in zip(pn, pc)}
    Pm["cap_n30"] = capv
    st = dict(tercile_edges_n_rigids=[float(t1), float(t2)], strata={})
    for name, (rm, sm) in defs.items():
        s = dict(n_slates=int(sm.sum()), n_rows=int(rm.sum()))
        for n in list(Pm):
            s[f"slateN_{n}"] = float(Pm[n][sm].mean())
            s[f"acc_{n}"] = acc(err[n][rm], ep[rm])
        for b in ["gnn_old_n30", "gnn_old_n200", "nfd_v2_s0", "lf_v2_switched"]:
            d = Pm["gnn_v2_n30"][sm] - Pm[b][sm]
            s[f"slateN_delta gnn_v2_n30 - {b}"] = dict(mean=float(d.mean()), ci=boot_mean_ci(d))
        st["strata"][name] = s
    out["strata"] = st
    out["degeneracy_frac_dv_true_zero"] = {g: {v: d61["degeneracy"][g][v]["frac_dv_true_zero"] for v in VFS} for g in GOALS}
    _save_json("gnn_ds0019.json", out)
    write_md(out)


def write_md(o):
    L = ["# EXP-0062 RUN-0003 -- dyn-res GNN trained on DS-0020 v2, constant 30 nodes (MODEL-0010), on DS-0019", "",
         f"Truth: {o['truth']} (values {o['truth_check']['unique_values']}). {o['n_slates']} slates, {o['n_rows']} rows.",
         f"Reuse check (original ckpt N 200 re-scored vs EXP-0061 `gnn_flex_drp_samp0`): "
         f"max |per-slate diff| {o['reuse_check_old_n200']['max_abs_per_slate_diff']:.3g}, max |row rms diff| "
         f"{o['reuse_check_old_n200']['max_abs_row_rms_diff']:.3g}, accuracy {o['reuse_check_old_n200']['accuracy_old']:.4f} -> "
         f"{o['reuse_check_old_n200']['accuracy_new']:.4f}.",
         f"Small-pile fallback (states with <= N voxels): " + ", ".join(f"{k} {v['states']}/{v['of']}" for k, v in o["fallback"].items())
         + f"; cap {o['cap_fallback']['states']}/{o['cap_fallback']['of']}.", "",
         "## slateN, mean of 3 goals [slate-bootstrap 95% CI]", "",
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
          "| model | DS-0019 | slate CI | DS-0020 v2 val | val traj CI |", "|---|---|---|---|---|"]
    for m in [k for k in a if not k.startswith("delta") and k != "persistence"]:
        v = va.get(m) or va.get(m + "_rescored") or {}
        vci = v.get("traj_ci")
        L.append(f"| {m} | {a[m]['accuracy']:.4f} | {a[m]['slate_ci'][0]:.3f}-{a[m]['slate_ci'][1]:.3f} | "
                 f"{v.get('accuracy', float('nan')):.4f} | " + (f"{vci[0]:.3f}-{vci[1]:.3f}" if vci else "") + " |")
    L.append("| persistence | 0 | | 0 | |")
    L += [""] + [f"- {k}: {v['delta']:+.4f} [{v['slate_ci'][0]:+.4f}, {v['slate_ci'][1]:+.4f}] (DS-0019, slate-cluster)"
                 for k, v in a.items() if k.startswith("delta")]
    L += [f"- GNN v2 accuracy / rendering cap (true node motion, N 30) = {o['dynamics_share_of_cap']:.3f}",
          f"- slateN cap - GNN v2 (all 9): {o['slateN_cap_minus_gnn_v2']['mean']:+.4f} {o['slateN_cap_minus_gnn_v2']['ci']}",
          f"- DS-0020 v2 val rows: {va['n_rows']}; val fallback states: {va['fallback']}", "",
          "## paired slateN (paired_stats.paired_comparison; per-slate mean of the cells; Holm over all pairs of the listed models, GNN-v2 pairs shown)", "",
          "| cells | a - b | diff [95% CI] | wins/losses | Holm p |", "|---|---|---|---|---|"]
    for lab, prs in list(o["paired_slateN"].items()) + [(f"goal {g}", p) for g, p in o["paired_slateN_per_goal"].items()]:
        for p in prs:
            L.append(f"| {lab} | {p['a']} - {p['b']} | {p['mean_diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['wins_a']}/{p['wins_b']} | {p['p_holm']:.2g} |")
    L += ["", f"## strata (piece-count tercile edges {o['strata']['tercile_edges_n_rigids']}; slateN = per-slate mean of 9 cells)", "",
          "| stratum (slates) | slateN GNN v2 / old N30 / old N200 / NFD v2 / LF v2 / cap | d(v2 - old N30) | d(v2 - old N200) | d(v2 - NFD v2) | d(v2 - LF v2) | acc GNN v2 / cap |",
          "|---|---|---|---|---|---|---|"]
    for n, s in o["strata"]["strata"].items():
        ds = [s[f"slateN_delta gnn_v2_n30 - {b}"] for b in ["gnn_old_n30", "gnn_old_n200", "nfd_v2_s0", "lf_v2_switched"]]
        L.append(f"| {n} ({s['n_slates']}) | " + " / ".join(f"{s['slateN_' + m]:.3f}" for m in
                 ["gnn_v2_n30", "gnn_old_n30", "gnn_old_n200", "nfd_v2_s0", "lf_v2_switched", "cap_n30"]) + " | "
                 + " | ".join(f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" for d in ds)
                 + f" | {s['acc_gnn_v2_n30']:.3f} / {s['acc_cap_n30']:.3f} |")
    tmp = R / "gnn_ds0019.md.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, R / "gnn_ds0019.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["score", "analyze"])
    ap.add_argument("--alt-ckpt", default=None, help="the non-primary-target GNN (sensitivity row)")
    ap.add_argument("--resume", action="store_true", help="skip DS-0019 models already in the raw file")
    ap.add_argument("--only", default=None, help="comma list of model names to score now")
    ap.add_argument("--skip-val-old", action="store_true", help="skip the N=200 original ckpt on val (slow)")
    a = ap.parse_args()
    score(a) if a.what == "score" else analyze(a)
