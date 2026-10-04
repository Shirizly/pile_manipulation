"""EXP-0064 RUN-0015 -- image-metric scoring on DS-0022 for the image-based families (NFD, switched
LinearForesight) plus the EXP-0064 particle GNN carried onto the image mask, with baselines.

Everything is on the FlexData / eval_report image-mask path (the EXP-0061/0062 pipeline):
  cell   = FlexData.load_flex_cell(DS-0022 config, "all", occ_source="image_mask")
           (rows flagged nan / escaped (|coord| > 10) / out_of_grid / null are dropped by the loader)
  occ0   = binary top-down colour-image mask before the push (model input)
  truth  = cell.occ1 = binary image mask after the push (= eval_report --truth-scoring image)
  accuracy (per row): swept-region rms (EXP-0061 final_eval.row_rms = metrics()'s per-row term);
           any set of rows: 1 - sum(rms_model) / sum(rms_persistence)  (= fit_linear_foresight.metrics)
  slateN values: per row, goal x value fn (random_quadrant seeded by state / ring_O / T  x
           lyapunov / mass_in_region / signed_mass), exactly eval_report._goals_for_slate + the
           Baselines.common.goals value functions. Captures are computed in `analyze`.

GNN on the image (models gnn12, field): MODEL-0012 (action z sign -1, tube encoding) predicts its
<= 30 FPS node displacements from the PARTICLE state (EXP-0064 eval_extended.py rep 0: FPS seed
1000*0 + state_idx, node budget min(30, target count)); every occupied input-mask cell is
supersampled 4x4 and each sub-point carried by its nearest node's table-frame displacement, then
binned onto the grid with floor(q*to_pxl + ctr + 0.5) (Baselines/GNN/flex_predictor.py::render's
rule). `gnn_truecap` = the same carry with the TRUE displacement of the same FPS nodes (cap of
the node-carry renderer, as EXP-0062 eval_gnn_v2.true_motion_cap). `field` = the same carry with the untrained action field s0 + build_action_delta(-1) (EXP-0064's
`field` baseline). Zero displacement reproduces occ0 exactly (asserted).

    python -u .../score_image_metrics.py score --models persistence,random,lf13_switched,...
    python -u .../score_image_metrics.py analyze
Per-model outputs: artifacts/RUN-0015-image-metric-scoring/<model>.npz (atomic, skip if present).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(REPO.glob("experiments/EXP-0061-*/code"))))
os.chdir(REPO)

E = REPO / "experiments/EXP-0064-obj-count-effect-study"
ART = E / "artifacts/RUN-0015-image-metric-scoring"
RES = E / "results"
D22 = "datasets/DS-0022-flex-carrots-countgroups-test-slates/config.yaml"
D22_DATA = REPO / "datasets/DS-0022-flex-carrots-countgroups-test-slates/data"
GROUPS = ["10-30", "50-70", "100-150", "400-500"]
GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
HIB = {"lyapunov": False, "mass_in_region": True, "signed_mass": True}

MODEL_SPECS = {   # name -> (eval_report MODELS key, ckpt) for registered predictors
    "lf13_switched": ("lf_flex_switched", "weights/MODEL-0013-linear-foresight-flex-mask-countgroups/checkpoint.pt"),
    "lf13_single": ("lf_flex_single", "weights/MODEL-0013-linear-foresight-flex-mask-countgroups/checkpoint.pt"),
    "nfd14": ("nfd_randlen", "weights/MODEL-0014-nfd-flex-mask-countgroups-seed0/checkpoint.pth"),
    # out-of-domain references: trained on DS-0020 v2 (EXP-0062), same scene/grid, different piles
    "lf09_switched_ds0020": ("lf_flex_switched", "weights/MODEL-0009-linear-foresight-flex-mask-v2/checkpoint.pt"),
    "nfd08_ds0020": ("nfd_randlen", "weights/MODEL-0008-nfd-flex-mask-v2-seed0/checkpoint.pth"),
}
GNN12 = "weights/MODEL-0012-gnn-flex-particles-countgroups-fixed-frame/checkpoint.pth"
GNN_CFG = E / "code/configs/gnn_dyn_grouped.yaml"


def _atomic_npz(path: Path, **a):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **a)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


_CELL = None


def get_cell():
    global _CELL
    if _CELL is None:
        from FlexData.dataset import load_flex_cell
        _CELL = load_flex_cell(D22, "all", tag="ds0022_mask", occ_source="image_mask")
        o1 = _CELL.occ1
        assert set(torch.unique(o1).tolist()) <= {0.0, 1.0}, "truth is not a binary mask"
    return _CELL


def row_values(cell, img: torch.Tensor) -> np.ndarray:
    """(N, 9) slateN values [goal x vf] of images `img`, eval_report's goals/value fns."""
    from Baselines.common import eval_report as er
    from Baselines.common.goals import mass_in_region, signed_mass_in_region
    from control_utility_test import lyapunov
    H, W = cell.occ0.shape[-2:]
    out = np.zeros((len(img), 9), np.float64)
    cache = {}
    sl = cell.slate_idx
    for sid in sl.unique().tolist():
        rows = (sl == sid).nonzero(as_tuple=True)[0]
        x = img[rows].float()
        for gi, (g, mask, dist) in enumerate(er._goals_for_slate("default", H, W, sid, cache)):
            assert g == GOALS[gi]
            out[rows.numpy(), 3 * gi + 0] = lyapunov(x, dist).double().numpy()
            out[rows.numpy(), 3 * gi + 1] = mass_in_region(x, mask).double().numpy()
            out[rows.numpy(), 3 * gi + 2] = signed_mass_in_region(x, mask).double().numpy()
    return out


def meta(cell):
    raw = cell.raw
    src_rows = raw._index_map
    aidx = raw.src.action_idx[src_rows]
    st = cell.slate_idx.numpy()
    split = json.loads((REPO / "datasets/DS-0022-flex-carrots-countgroups-test-slates/splits.json").read_text())
    grp = np.array([split["count_group"][str(int(s))] for s in st])
    return st, aidx, grp


def ensure_truth():
    f = ART / "_truth.npz"
    if f.exists():
        return
    from final_eval import region_of, row_rms
    cell = get_cell()
    region = region_of(cell)
    st, aidx, grp = meta(cell)
    truth, prev = cell.occ1.float(), cell.occ0.float()
    fl = cell.raw.flags
    _atomic_npz(f, state=st, action_idx=aidx, group=grp,
                rms_persistence=row_rms(prev, truth, region).numpy(),
                values_true=row_values(cell, truth), push_length=cell.push_length.numpy(),
                n_flag_escaped=int(fl["escaped"].sum()), n_flag_oog=int(fl["out_of_grid"].sum()),
                n_flag_null=int(fl["null"].sum()), n_flag_nan=int(fl["nan"].sum()),
                region_px=region.reshape(len(region), -1).sum(1).numpy())
    print(f"truth: {len(st)} rows, {len(np.unique(st))} states", flush=True)


# ---------------------------------------------------------------- GNN carried onto the mask
def gnn_predict_images(cell, which: str, device: str = "cpu") -> torch.Tensor:
    import yaml
    from dataset_grouped_particles import GLOBAL_SCALE, PARTICLE_DEN_CONST, _fps_indices, encode_action
    from Baselines.GNN.model.gnn_dyn import PropNetDiffDenModel
    dev = torch.device(device)
    model = None
    if which == "gnn12":
        cfg = yaml.safe_load(open(GNN_CFG))
        model = PropNetDiffDenModel(cfg, dev.type == "cuda")
        model.load_state_dict(torch.load(GNN12, map_location=dev))
        model = model.to(dev).eval()
    st, aidx, _ = meta(cell)
    recs = [json.loads(l) for l in open(D22_DATA / "manifest.jsonl")]
    init = {r["state_idx"]: r for r in recs if r["type"] == "state_init"}
    act = {(r["state_idx"], r["action_idx"]): r["action"] for r in recs if r["type"] == "action"}
    H, W = cell.H, cell.W
    to_pxl, ctr = float(cell.raw.to_pxl), float(cell.raw.ctr_in_PXL[0])
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    out = torch.zeros(len(st), H, W)
    t0 = time.time()
    for s in np.unique(st):
        rows = np.nonzero(st == s)[0]
        rec0 = init[int(s)]
        p0 = np.load(D22_DATA / rec0["positions_path"]).reshape(-1, 4)[:, :3]
        nb = min(30, int(rec0.get("target_num_carrots") or 30))
        np.random.seed(1000 * 0 + int(s))                               # eval_extended rep 0
        idx = _fps_indices(p0 / GLOBAL_SCALE, nb)
        n = len(idx)
        s0 = (p0[idx] / GLOBAL_SCALE)[:, [0, 2, 1]].astype(np.float32)     # (n,3) normalised (x, z, y)
        A = np.array([act[(int(s), int(a))] for a in aidx[rows]], np.float32)
        B = len(rows)
        sd = torch.zeros((B, n, 3))
        for bi in range(B):
            a = A[bi] / GLOBAL_SCALE
            sd[bi] = encode_action("tube", torch.from_numpy(s0), torch.tensor([a[0], -a[1], 0.0]),
                                   torch.tensor([a[2], -a[3], 0.0]))
        if which == "gnn12":
            with torch.no_grad():
                pr = model.predict_one_step(torch.ones((B, n), device=dev),
                                            torch.from_numpy(np.tile(s0[None], (B, 1, 1))).to(dev),
                                            sd.to(dev), torch.full((B,), PARTICLE_DEN_CONST, device=dev)).cpu().numpy()
        elif which == "field":
            pr = s0[None] + sd.numpy()
        elif which == "gnn_truecap":   # renderer cap: TRUE node motion (tracked particles' after-positions)
            ap = {r["action_idx"]: r["after_positions_path"] for r in recs
                  if r["type"] == "action" and r["valid"] and r["state_idx"] == int(s)}
            pr = np.stack([(np.load(D22_DATA / ap[int(a)]).reshape(-1, 4)[idx, :3] / GLOBAL_SCALE)[:, [0, 2, 1]]
                           for a in aidx[rows]]).astype(np.float32)
        else:  # zero-displacement self-check
            pr = np.tile(s0[None], (B, 1, 1))
        d = (pr - s0[None])[..., :2] * GLOBAL_SCALE                        # (B,n,2) flex (dx, dz)
        d_tab = np.stack([d[..., 0], -d[..., 1]], -1)                      # table (dX, dY)
        nodes_tab = np.stack([p0[idx, 0], -p0[idx, 2]], 1)                 # (n,2)
        occ0 = cell.occ0[rows[0]].numpy() > 0.5
        ii, jj = np.nonzero(occ0)
        pi = (ii[:, None, None] + sub[None, :, None]).repeat(4, 2).reshape(len(ii), -1)
        pj = (jj[:, None, None] + sub[None, None, :]).repeat(4, 1).reshape(len(jj), -1)
        P = np.stack([(pi.ravel() - ctr) / to_pxl, (pj.ravel() - ctr) / to_pxl], 1)   # world table XY
        nn = ((P[:, None] - nodes_tab[None]) ** 2).sum(-1).argmin(1)
        Q = P[None] + d_tab[:, nn]                                          # (B, M, 2)
        gi = np.floor(Q[..., 0] * to_pxl + ctr + 0.5).astype(int)
        gj = np.floor(Q[..., 1] * to_pxl + ctr + 0.5).astype(int)
        ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
        bi = np.broadcast_to(np.arange(B)[:, None], gi.shape)
        img = np.zeros((B, H * W), np.float32)
        img[bi[ok], gi[ok] * W + gj[ok]] = 1.0
        out[torch.from_numpy(rows)] = torch.from_numpy(img.reshape(B, H, W))
    print(f"[{which}] carried {len(np.unique(st))} states in {time.time() - t0:.0f}s", flush=True)
    return out


def predict(name: str, cell, device: str) -> torch.Tensor:
    if name == "persistence":
        return cell.occ0.float().clone()
    if name == "random":
        return torch.rand(cell.occ0.shape, generator=torch.Generator().manual_seed(0))
    if name in ("gnn12", "field", "gnn_truecap"):
        return gnn_predict_images(cell, name, device)
    from Baselines.common import eval_report as er
    key, ck = MODEL_SPECS[name]
    spec = dict(er.MODELS[key], ckpt=ck)
    assert Path(ck).exists(), ck
    return er._predict(er._load_predictor(spec), cell, device)


def score(args):
    from final_eval import region_of, row_rms
    ART.mkdir(parents=True, exist_ok=True)
    ensure_truth()
    cell = get_cell()
    region = region_of(cell)
    truth = cell.occ1.float()
    if args.self_check:
        z = gnn_predict_images(cell, "zero", "cpu")
        assert torch.equal(z, cell.occ0.float()), "zero-displacement carry does not reproduce occ0"
        print("self-check: zero-displacement carry == occ0", flush=True)
    for name in args.models.split(","):
        f = ART / f"{name}.npz"
        if f.exists():
            print(f"[{name}] exists, skip", flush=True)
            continue
        t0 = time.time()
        pred = predict(name, cell, args.device).float()
        assert pred.shape == truth.shape
        _atomic_npz(f, rms=row_rms(pred, truth, region).numpy(), values=row_values(cell, pred),
                    seconds=time.time() - t0, device=args.device,
                    pred_u8=(pred.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        acc = 1 - float(np.load(f)["rms"].sum() / np.load(ART / "_truth.npz")["rms_persistence"].sum())
        print(f"[{name}] accuracy {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- analysis
def capture_pool(vp: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """vp, vt (..., n, 9) oriented higher-is-better -> (..., 9) slateN capture; ties in the model's
    value are broken uniformly at random IN EXPECTATION (mean true value over the tied argmax set),
    so a constant predictor (persistence) scores exactly 0. NaN where best == mean (degenerate)."""
    mx = vp.max(-2, keepdims=True)
    tie = np.isclose(vp, mx, rtol=1e-6, atol=1e-9)
    chosen = (vt * tie).sum(-2) / tie.sum(-2)
    mean, best = vt.mean(-2), vt.max(-2)
    den = best - mean
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9 * (np.abs(best) + 1), (chosen - mean) / den, np.nan)


def orient(v):
    s = np.array([-1.0 if not HIB[vf] else 1.0 for g in GOALS for vf in VFS])
    return v * s


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 2 else float("nan")


def analyze(args):
    from scipy.stats import kendalltau, spearmanr
    rng = np.random.default_rng(0)
    T = dict(np.load(ART / "_truth.npz"))
    st, grp = T["state"], T["group"]
    states = np.unique(st)
    sgrp = np.array([grp[st == s][0] for s in states])
    rows_of = [np.nonzero(st == s)[0] for s in states]
    pool = np.array([len(r) for r in rows_of])
    K = int(pool.min()) if args.k is None else args.k
    vt = orient(T["values_true"])
    models = [m for m in args.order.split(",") if (ART / f"{m}.npz").exists()]
    NSUB = args.n_sub
    subsets = [np.stack([rng.choice(len(r), K, replace=False) for _ in range(NSUB)]) for r in rows_of]
    per_state = {}
    for m in models:
        Z = np.load(ART / f"{m}.npz")
        vp = orient(Z["values"])
        cap_full = np.stack([capture_pool(vp[r], vt[r]) for r in rows_of])                       # (S, 9)
        cap_K = np.stack([np.nanmean(capture_pool(vp[r][sub], vt[r][sub]), 0) for r, sub in zip(rows_of, subsets)])
        rms = Z["rms"]
        num = np.array([rms[r].sum() for r in rows_of]); den = np.array([T["rms_persistence"][r].sum() for r in rows_of])
        per_state[m] = dict(num=num, den=den, acc=1 - num / den,
                            slateN=np.nanmean(cap_full, 1), slateN_K=np.nanmean(cap_K, 1),
                            cap_full=cap_full, cap_K=cap_K, seconds=float(Z["seconds"]), device=str(Z["device"]))
    degenerate = np.stack([np.isnan(capture_pool(vt[r], vt[r])) for r in rows_of])               # (S, 9)

    def cell_stats(m, sel, nb=args.n_boot):
        d = per_state[m]
        idx = np.nonzero(sel)[0]
        acc = 1 - d["num"][idx].sum() / d["den"][idx].sum()
        B = rng.integers(0, len(idx), (nb, len(idx)))
        jj = idx[B]
        accb = 1 - d["num"][jj].sum(1) / d["den"][jj].sum(1)
        o = dict(n_states=int(len(idx)), accuracy=float(acc), accuracy_ci=np.quantile(accb, [.025, .975]).tolist())
        for k in ("slateN", "slateN_K"):
            x = d[k][idx]
            o[k] = float(np.nanmean(x)); o[k + "_ci"] = np.nanquantile(np.nanmean(x[B], 1), [.025, .975]).tolist()
        return o

    sels = {g: sgrp == gi for gi, g in enumerate(GROUPS)}
    sels["overall"] = np.ones(len(states), bool)
    res = dict(K_eq=K, n_sub=NSUB, n_boot=args.n_boot, models=models,
               truth="binary image mask of the after-state (cell.occ1, occ_source=image_mask) = eval_report --truth-scoring image",
               pool={g: dict(mean=float(pool[s].mean()), min=int(pool[s].min()), max=int(pool[s].max()))
                     for g, s in sels.items()},
               degenerate_frac={g: float(degenerate[s].mean()) for g, s in sels.items()},
               flags=dict(escaped=int(T["n_flag_escaped"]), out_of_grid=int(T["n_flag_oog"]), null=int(T["n_flag_null"]),
                          nan=int(T["n_flag_nan"]), kept_rows=int(len(st))),
               cells={m: {g: cell_stats(m, s) for g, s in sels.items()} for m in models},
               runtime={m: dict(seconds=per_state[m]["seconds"], device=per_state[m]["device"]) for m in models})
    # per goal x vf (overall + per group, K-eq) for the record
    res["per_goal_vf_Keq"] = {m: {g: dict(zip([f"{a}/{b}" for a in GOALS for b in VFS],
                                              np.nanmean(per_state[m]["cap_K"][s], 0).round(4).tolist()))
                                  for g, s in sels.items()} for m in models}
    # paired differences (K-eq slateN, accuracy) vs chosen references
    pairs = [(a, b) for a, b in [("nfd14", "lf13_switched"), ("nfd14", "gnn12"), ("lf13_switched", "gnn12"),
                                 ("gnn12", "field"), ("nfd14", "field"), ("lf13_switched", "field"),
                                 ("nfd14", "nfd08_ds0020"), ("lf13_switched", "lf09_switched_ds0020")]
             if a in models and b in models]
    res["paired"] = {}
    for a, b in pairs:
        res["paired"][f"{a} - {b}"] = {}
        for g, s in sels.items():
            idx = np.nonzero(s)[0]
            B = idx[rng.integers(0, len(idx), (args.n_boot, len(idx)))]
            dS = per_state[a]["slateN_K"] - per_state[b]["slateN_K"]
            A, Bm = per_state[a], per_state[b]
            accd = lambda j: (1 - A["num"][j].sum(-1) / A["den"][j].sum(-1)) - (1 - Bm["num"][j].sum(-1) / Bm["den"][j].sum(-1))
            res["paired"][f"{a} - {b}"][g] = dict(
                slateN_K=float(np.nanmean(dS[idx])), slateN_K_ci=np.nanquantile(np.nanmean(dS[B], 1), [.025, .975]).tolist(),
                accuracy=float(accd(idx)), accuracy_ci=np.quantile(accd(B), [.025, .975]).tolist())
    # group contrast 400-500 minus 10-30 (unpaired, state bootstrap)
    res["contrast_400_minus_10"] = {}
    i0, i3 = np.nonzero(sels["10-30"])[0], np.nonzero(sels["400-500"])[0]
    for m in models:
        d = per_state[m]
        b0 = i0[rng.integers(0, len(i0), (args.n_boot, len(i0)))]; b3 = i3[rng.integers(0, len(i3), (args.n_boot, len(i3)))]
        a = lambda j: 1 - d["num"][j].sum(-1) / d["den"][j].sum(-1)
        res["contrast_400_minus_10"][m] = dict(
            accuracy=float(a(i3) - a(i0)), accuracy_ci=np.quantile(a(b3) - a(b0), [.025, .975]).tolist(),
            slateN_K=float(np.nanmean(d["slateN_K"][i3]) - np.nanmean(d["slateN_K"][i0])),
            slateN_K_ci=np.nanquantile(np.nanmean(d["slateN_K"][b3], 1) - np.nanmean(d["slateN_K"][b0], 1), [.025, .975]).tolist())
    # ---- correlations
    cor = {}
    # (a) across models within each group (+overall)
    for label, ms in (("all_incl_baselines", models),
                      ("trained_and_field_only", [m for m in models if m not in ("persistence", "random")])):
        cor.setdefault("a_across_models", {})[label] = {}
        for g in sels:
            for key in ("slateN_K", "slateN"):
                x = np.array([res["cells"][m][g]["accuracy"] for m in ms])
                y = np.array([res["cells"][m][g][key] for m in ms])
                kt = kendalltau(x, y); sp = spearmanr(x, y)
                cor["a_across_models"][label].setdefault(g, {})[key] = dict(
                    n_models=len(ms), models=ms, kendall_tau=float(kt.statistic), kendall_p=float(kt.pvalue),
                    spearman=float(sp.statistic), spearman_p=float(sp.pvalue))
    # (b) across groups within each model (n = 4)
    cor["b_across_groups"] = {}
    for m in models:
        x = np.array([res["cells"][m][g]["accuracy"] for g in GROUPS])
        y = np.array([res["cells"][m][g]["slateN_K"] for g in GROUPS])
        cor["b_across_groups"][m] = dict(n_groups=4, spearman=spearman(x, y), kendall_tau=float(kendalltau(x, y).statistic),
                                         acc_rank=np.argsort(np.argsort(-x)).tolist(), slateN_K_rank=np.argsort(np.argsort(-y)).tolist())
    # (c) per state within each model: Spearman(state accuracy, state slateN), bootstrap over states
    cor["c_per_state"] = {}
    for m in models:
        if m in ("persistence",):
            continue
        d = per_state[m]
        cor["c_per_state"][m] = {}
        for g, s in sels.items():
            idx = np.nonzero(s)[0]
            o = {}
            for key in ("slateN", "slateN_K"):
                rho = spearman(d["acc"][idx], d[key][idx])
                bs = [spearman(d["acc"][j], d[key][j]) for j in idx[rng.integers(0, len(idx), (args.n_boot_rho, len(idx)))]]
                o[key] = dict(n_states=int(len(idx)), spearman=rho, ci=np.nanquantile(bs, [.025, .975]).tolist())
            cor["c_per_state"][m][g] = o
    res["correlations"] = cor
    RES.mkdir(exist_ok=True)
    _atomic_json(RES / "nfd_lf_image_metrics.json", res)
    np.savez(ART / "_per_state.npz", states=states, group=sgrp, pool=pool,
             **{f"{m}__{k}": per_state[m][k] for m in models for k in ("acc", "slateN", "slateN_K")})
    write_md(res)
    print(f"analyze: {models} K={K}", flush=True)


def write_md(res):
    f = lambda c, k: f"{c[k]:+.3f} [{c[k + '_ci'][0]:+.3f}, {c[k + '_ci'][1]:+.3f}]"
    L = ["# EXP-0064 RUN-0015 -- image-metric scoring on DS-0022 (generated by code/score_image_metrics.py analyze)", "",
         f"Truth: {res['truth']}. K_eq = {res['K_eq']} (smallest clean pool), {res['n_sub']} random K-subsets per state. "
         f"Cells: mean [95% state-bootstrap CI, {res['n_boot']} reps]; accuracy = 1 - sum rms / sum rms_persistence "
         f"(swept region, eval_report definition). slateN = mean over 3 goals x 3 value fns; ties in a model's value "
         f"broken uniformly in expectation (persistence = 0 exactly). random = i.i.d. uniform noise image (seed 0).", "",
         "## Pools and flags", "", "| group | pool mean | min | max | degenerate goal-vf frac |", "|---|---|---|---|---|"]
    for g, p in res["pool"].items():
        L.append(f"| {g} | {p['mean']:.1f} | {p['min']} | {p['max']} | {res['degenerate_frac'][g]:.3f} |")
    L += ["", f"Loader flags (rows dropped): {res['flags']}", "", "## Per group: slateN at K_eq (lead), slateN full pool, accuracy", ""]
    for key, title in (("slateN_K", "slateN, K-equalised"), ("slateN", "slateN, full clean pool"), ("accuracy", "accuracy (swept region; suspect across model types)")):
        L += [f"### {title}", "", "| model | " + " | ".join(GROUPS + ["overall"]) + " |", "|---|" + "---|" * 5]
        for m, c in res["cells"].items():
            L.append(f"| {m} | " + " | ".join(f(c[g], key) for g in GROUPS + ["overall"]) + " |")
        L.append("")
    L += ["## Paired differences (per state)", "", "| pair | group | slateN_K | accuracy |", "|---|---|---|---|"]
    for p, d in res["paired"].items():
        for g, c in d.items():
            L.append(f"| {p} | {g} | {f(c, 'slateN_K')} | {f(c, 'accuracy')} |")
    L += ["", "## 400-500 minus 10-30 (unpaired)", "", "| model | accuracy | slateN_K |", "|---|---|---|"]
    for m, c in res["contrast_400_minus_10"].items():
        L.append(f"| {m} | {f(c, 'accuracy')} | {f(c, 'slateN_K')} |")
    cor = res["correlations"]
    L += ["", "## Correlation of accuracy and slateN", "", "### (a) across models, within group", "",
          "| model set | group | n | Kendall tau (p) slateN_K | Spearman (p) slateN_K | Kendall tau (p) slateN full |", "|---|---|---|---|---|---|"]
    for lab, d in cor["a_across_models"].items():
        for g, c in d.items():
            a, b = c["slateN_K"], c["slateN"]
            L.append(f"| {lab} | {g} | {a['n_models']} | {a['kendall_tau']:+.2f} ({a['kendall_p']:.3f}) | "
                     f"{a['spearman']:+.2f} ({a['spearman_p']:.3f}) | {b['kendall_tau']:+.2f} ({b['kendall_p']:.3f}) |")
    L += ["", "### (b) across the 4 groups, within model (n = 4; |rho| = 1 is the only two-sided p < 0.1)", "",
          "| model | Spearman | Kendall | accuracy rank by group | slateN_K rank by group |", "|---|---|---|---|---|"]
    for m, c in cor["b_across_groups"].items():
        L.append(f"| {m} | {c['spearman']:+.2f} | {c['kendall_tau']:+.2f} | {c['acc_rank']} | {c['slateN_K_rank']} |")
    L += ["", "### (c) per state, within model: Spearman(state accuracy, state slateN) [95% CI]", "",
          "| model | slate | " + " | ".join(GROUPS + ["overall"]) + " |", "|---|---|" + "---|" * 5]
    for m, d in cor["c_per_state"].items():
        for key in ("slateN_K", "slateN"):
            L.append(f"| {m} | {key} | " + " | ".join(
                f"{d[g][key]['spearman']:+.2f} [{d[g][key]['ci'][0]:+.2f}, {d[g][key]['ci'][1]:+.2f}] (n={d[g][key]['n_states']})"
                for g in GROUPS + ["overall"]) + " |")
    L += ["", "## Runtime", "", "| model | seconds | device |", "|---|---|---|"]
    for m, c in res["runtime"].items():
        L.append(f"| {m} | {c['seconds']:.0f} | {c['device']} |")
    (RES / "nfd_lf_image_metrics.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("score")
    a.add_argument("--models", required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--self-check", action="store_true")
    b = sp.add_parser("analyze")
    b.add_argument("--order", default="nfd14,lf13_switched,lf13_single,gnn12,gnn_truecap,field,nfd08_ds0020,lf09_switched_ds0020,persistence,random")
    b.add_argument("--k", type=int, default=None)
    b.add_argument("--n-sub", type=int, default=50)
    b.add_argument("--n-boot", type=int, default=2000)
    b.add_argument("--n-boot-rho", type=int, default=1000)
    args = ap.parse_args()
    score(args) if args.cmd == "score" else analyze(args)


if __name__ == "__main__":
    main()
