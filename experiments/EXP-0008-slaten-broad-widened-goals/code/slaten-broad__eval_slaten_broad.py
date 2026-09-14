"""Widen slateN validation along 4 axes (task: slaten-broad).

1. Re-pool slates_multistep EXCLUDING n20_L10mm entirely (user ruled it
   problematic) -- re-examine whether switched-vs-global still reverses
   once L10mm's weak-bin contamination is removed.
2. Score goal SHAPES beyond corner/center: stripe, random_quadrant, ring_O,
   T (Baselines/common/goals.py's own masks), each reported separately
   (never shape-averaged) with a frac(dv_true==0) degeneracy check per cell.
3. Two NEW goal FUNCTIONS beyond Lyapunov distance: mass_in_region and
   signed_mass_in_region (Baselines/common/goals.py, formulas already in
   experiments/METRICS.md's "generalised to an arbitrary VALUE function"
   section -- reused verbatim, not reinvented).
4. The overnight_randlen HELD-OUT file split (43/213 files never trained
   on) as an additional eval source, built into same-state slates directly
   from experiments/temp/hybrid-vis-desc/cache/test_cache.pt (verified
   below to be exactly the `descriptors.py::split_files(seed=0,
   holdout_frac=0.2)` test set stage2/stage3 operators were fit without).

Cells: stage-2 visual (switched / global unswitched), stage-3
latent+desc94 (switched / global), descriptor-only (point-mass value
readout, extended to all 3 value functions -- see `desc_pointmass_value`),
persistence, random floor. All operators REUSED from disk, never refit.

Reuses unchanged: `Baselines/common/goals.py` (mask/value-function
formulas, `slate_n_capture`), `control_utility_test.lyapunov`/
`lyapunov_weights`, stage-2's `fit_hybrid.predict_image`, stage-3's
`fit_latent_switched.predict_image`, `descriptors_d.slices_d`,
`eval_slaten_latent.py`'s `com_world_pixel`/`bilinear_sample`.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/latent-switched")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")

from control_utility_test import lyapunov, lyapunov_weights           # noqa: E402
from fit_linear_foresight import actions_to_pixels, canonicalise      # noqa: E402
from descriptors_d import push_frame_full_descriptors, slices_d       # noqa: E402
from descriptors_b import world_to_pushframe_px                       # noqa: E402
from descriptors import BOUNDS, GRID, list_files, split_files         # noqa: E402
from Baselines.common.data import load_cell                          # noqa: E402
from Baselines.common.goals import (                                  # noqa: E402
    slate_n_capture, mass_in_region, signed_mass_in_region,
    quadrant_mask, random_quadrant_mask, letter_mask, dist_field_from_mask,
)
from fit_hybrid import CELLS as CELLS2, bin_index, predict_image as predict_image2  # noqa: E402
from train_encoder import Encoder, Decoder                            # noqa: E402
from fit_latent_switched import predict_image as predict_image3       # noqa: E402
from eval_slaten_latent import com_world_pixel, bilinear_sample       # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/slaten-broad"
STAGE2_OPDIR = "/home/alon/Code/pile_manipulation/experiments/temp/stage2-slaten/operators"
STAGE3_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten"
HYBRID_CACHE = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"

K_FIXED = 32
N_RESAMPLE = 150  # reduced from stage2/3's 300 for time budget; documented
DEGENERACY_THRESH = 0.90

WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])

MULTISTEP_DATASETS = {
    # n20_L10mm DELIBERATELY EXCLUDED (task instruction: ruled problematic).
    "n20_L20mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    "n20_L40mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
}

GOAL_SHAPES = ["corner", "stripe", "random_quadrant", "ring_O", "T"]
# 'center' checked-and-excluded already by stage2/stage3 RESULTS.md at this
# same degeneracy threshold on this same dataset family; not re-run here
# (would just reconfirm the same 99%+ zero result).
VALUE_FNS = ["lyapunov", "mass_in_region", "signed_mass_in_region"]


# ---------------------------------------------------------------------------
# Goal shape -> (binary mask, distance/weight field for lyapunov) builder
# ---------------------------------------------------------------------------

def build_goal(shape: str, H: int, W: int, slate_seed: int):
    """Returns (mask[bool,H,W], dw[float,H,W] for lyapunov cost)."""
    if shape in ("corner", "stripe"):
        dw = lyapunov_weights((H, W), shape, DEVICE)
        mask = np.zeros((H, W), dtype=bool)
        if shape == "corner":
            mask[: H // 2, : W // 2] = True
        else:  # stripe
            mask[H // 2 - H // 8: H // 2 + H // 8, :] = True
        mask = torch.from_numpy(mask)
    elif shape == "random_quadrant":
        mask_np, _q = random_quadrant_mask(H, W, seed=slate_seed)
        mask = torch.from_numpy(mask_np)
        dw = torch.from_numpy(dist_field_from_mask(mask_np)).to(DEVICE)
    elif shape == "ring_O":
        mask_np = letter_mask("O", H, W)
        mask = torch.from_numpy(mask_np)
        dw = torch.from_numpy(dist_field_from_mask(mask_np)).to(DEVICE)
    elif shape == "T":
        mask_np = letter_mask("T", H, W)
        mask = torch.from_numpy(mask_np)
        dw = torch.from_numpy(dist_field_from_mask(mask_np)).to(DEVICE)
    else:
        raise ValueError(shape)
    return mask, dw


def value_true_and_pred(value_fn, occ_true, occ_pred_dict, dw, mask):
    """occ_true/occ_pred_dict[name]: (N,H,W) tensors (world frame, on CPU).
    Returns dict name -> (N,) value (RAW V, not dV -- valid per METRICS.md
    since V0 is identical across all candidates of a same-state slate and
    cancels in slate_n_capture's formula; confirmed there algebraically)."""
    out_true = None
    out_pred = {}
    mask_dev = mask.to(DEVICE)
    if value_fn == "lyapunov":
        out_true = lyapunov(occ_true.to(DEVICE), dw).cpu()
        for name, occ in occ_pred_dict.items():
            out_pred[name] = lyapunov(occ.to(DEVICE), dw).cpu()
        higher_is_better = False
    elif value_fn == "mass_in_region":
        out_true = mass_in_region(occ_true.to(DEVICE), mask_dev).cpu()
        for name, occ in occ_pred_dict.items():
            out_pred[name] = mass_in_region(occ.to(DEVICE), mask_dev).cpu()
        higher_is_better = True
    elif value_fn == "signed_mass_in_region":
        out_true = signed_mass_in_region(occ_true.to(DEVICE), mask_dev).cpu()
        for name, occ in occ_pred_dict.items():
            out_pred[name] = signed_mass_in_region(occ.to(DEVICE), mask_dev).cpu()
        higher_is_better = True
    else:
        raise ValueError(value_fn)
    return out_true, out_pred, higher_is_better


def desc_pointmass_value(value_fn, world_row_px, world_col_px, mass_hat, dw, mask, H=64, W=64):
    """Point-mass (monopole) value readout for the descriptor-only model,
    extended from eval_slaten_latent.py's lyapunov-only V_hat = d(COM) to
    all 3 value functions:
      lyapunov:            V_hat = d(world_COM)                      (intensive; mass-normalised by construction, matches lyapunov()'s own /sum(occ))
      mass_in_region:      V_hat = mass_hat * 1[world_COM in mask]   (all predicted mass treated as concentrated at the predicted COM)
      signed_mass_in_region:V_hat = mass_hat * (2*1[world_COM in mask] - 1)
    `mass_hat` = model's predicted TOTAL pile mass (global_mass * H*W, world
    units matching sum(occ)), so mass_in_region/signed_mass_in_region are on
    the SAME scale as the true (raw-occupancy) value functions above.
    Uses bilinear sampling (not a hard index) at the (possibly off-grid)
    predicted COM, consistent with the existing lyapunov point-mass code."""
    if value_fn == "lyapunov":
        return bilinear_sample(dw.cpu(), world_row_px, world_col_px)
    mask_f = mask.float()
    m_at_com = bilinear_sample(mask_f, world_row_px, world_col_px)
    if value_fn == "mass_in_region":
        return mass_hat * m_at_com
    elif value_fn == "signed_mass_in_region":
        return mass_hat * (2 * m_at_com - 1)
    raise ValueError(value_fn)


# ---------------------------------------------------------------------------
# Loading persisted operators (never refit)
# ---------------------------------------------------------------------------

def load_stage2_visual():
    d = torch.load(f"{STAGE2_OPDIR}/visual_lam1.0.pt", map_location="cpu", weights_only=False)
    return d["cell_spec"], d["switched_ops"], d["global_op"], d["bin_edges"]


def load_stage3_latent():
    d = torch.load(f"{STAGE3_DIR}/latent_operators.pt", map_location="cpu", weights_only=False)
    E = Encoder(d["latent_dim"]).to(DEVICE)
    Dec = Decoder(d["latent_dim"]).to(DEVICE)
    E.load_state_dict(d["encoder"]); Dec.load_state_dict(d["decoder"])
    E.eval(); Dec.eval()
    for p in list(E.parameters()) + list(Dec.parameters()):
        p.requires_grad_(False)
    return E, Dec, d["latent_dim"], d["bin_edges"], d["ops_latent_desc94"], d["A_single_latent_desc94"]


def load_desc_operator():
    d = torch.load(f"{STAGE3_DIR}/desc_operator.pt", map_location="cpu", weights_only=False)
    return d["ops"], d["bin_edges"], d["slices"]


CELL_LD = dict(use_latent=True, desc_dim=94)


# ---------------------------------------------------------------------------
# Predict-cache builders
# ---------------------------------------------------------------------------

def build_cache_multistep(slates):
    occ0 = slates.occ0.float()
    H, W = occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(slates.actions, torch.tensor(slates.workspace_min),
                                    torch.tensor(slates.workspace_max), (H, W))
    RES = 32
    canon0 = canonicalise(occ0.to(DEVICE), s_px.to(DEVICE), e_px.to(DEVICE), RES, 1.0).cpu()
    N = occ0.shape[0]
    length_m = (slates.p_stop[:, :2] - slates.p_start[:, :2]).norm(dim=-1)
    gm0 = occ0.sum(dim=(-2, -1)) / (H * W)
    local0 = push_frame_full_descriptors(occ0.to(DEVICE), s_px.to(DEVICE), e_px.to(DEVICE)).cpu()
    desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)
    sp_d = world_to_pushframe_px(slates.p_start[:, :2])
    ep_d = world_to_pushframe_px(slates.p_stop[:, :2])
    return dict(occ0=occ0, canon0f=canon0.reshape(N, -1), start_px=s_px, end_px=e_px,
                length_m=length_m, desc0=desc0, sp_d=sp_d, ep_d=ep_d), canon0


def build_cache_holdout(test_cache, file_ids_holdout):
    """Group experiments/temp/hybrid-vis-desc/cache/test_cache.pt rows by
    file, take each file's FIRST 128 rows (step 0, shared-start-state pool
    -- verified: raw .pt files are 512 rows = 4 steps x 128 candidates,
    row//128 == step, rows within a step share one starting `states`
    tensor). Returns cache dict (same shape as build_cache_multistep) +
    slate_idx (=file_id) for the step-0 subset only."""
    fid = test_cache["file_id"]
    keep_rows = []
    slate_ids = []
    for f in file_ids_holdout:
        rows = (fid == f).nonzero(as_tuple=True)[0]
        assert rows.numel() == 512, f"file {f}: expected 512 rows, got {rows.numel()}"
        step0_rows = rows[:128]
        keep_rows.append(step0_rows)
        slate_ids.append(torch.full((128,), f, dtype=torch.long))
    idx = torch.cat(keep_rows)
    slate_idx = torch.cat(slate_ids)
    occ0 = test_cache["occ0"][idx].float()
    occ1 = test_cache["occ1"][idx].float()
    canon0 = test_cache["canon0"][idx].float()
    desc0 = test_cache["desc0"][idx]
    action = test_cache["action"][idx]
    length_m = test_cache["length_m"][idx]
    H, W = occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(action, WS_MIN, WS_MAX, (H, W))
    sp_d = world_to_pushframe_px(action[:, 0:2])
    ep_d = world_to_pushframe_px(action[:, 2:4])
    N = occ0.shape[0]
    cache = dict(occ0=occ0, canon0f=canon0.reshape(N, -1), start_px=s_px, end_px=e_px,
                 length_m=length_m, desc0=desc0, sp_d=sp_d, ep_d=ep_d)
    return cache, canon0, occ1, slate_idx


# ---------------------------------------------------------------------------
# Model predictions given a built cache
# ---------------------------------------------------------------------------

def predict_all_models(cache, canon0, E, Dec, latent_dim,
                        cell_v_spec, sw_ops_v, gl_op_v, be_v,
                        be_ld, ops_ld, A_ld,
                        desc_ops, be_desc, SD):
    occ0 = cache["occ0"]
    N = occ0.shape[0]

    # stage-2 visual: switched, global
    pred_v_sw = predict_image2(cell_v_spec, sw_ops_v, True, cache, be_v)
    pred_v_gl = predict_image2(cell_v_spec, gl_op_v, False, cache, be_v)

    # stage-3 latent+desc94: switched, global (needs z0 from frozen encoder)
    with torch.no_grad():
        z0_list = []
        for i in range(0, canon0.shape[0], 4096):
            z0_list.append(E(canon0[i:i + 4096].to(DEVICE)).cpu())
        z0 = torch.cat(z0_list, dim=0)
    cache_ld = dict(cache); cache_ld["z0"] = z0
    pred_ld_sw = predict_image3(CELL_LD, ops_ld, True, cache_ld, be_ld, Dec, latent_dim)
    pred_ld_gl = predict_image3(CELL_LD, A_ld, False, cache_ld, be_ld, Dec, latent_dim)

    # descriptor-only: point-mass readout (COM + total predicted mass)
    desc0 = cache["desc0"]
    length_m = cache["length_m"]
    bins_desc = bin_index(length_m, be_desc)
    phi1_hat = torch.empty_like(desc0)
    for b in range(len(desc_ops)):
        m = bins_desc == b
        if not bool(m.any()):
            continue
        A = desc_ops[b]
        phi1_hat[m] = (A @ desc0[m].T).T
    com_pred = phi1_hat[:, SD["com"]]
    mass_hat = phi1_hat[:, SD["global_mass"]][:, 0] * (occ0.shape[-2] * occ0.shape[-1])
    sp_d, ep_d = cache["sp_d"], cache["ep_d"]
    wr, wc = com_world_pixel(com_pred[:, 0], com_pred[:, 1], sp_d, ep_d)

    preds_img = {"visual_switched": pred_v_sw, "visual_global": pred_v_gl,
                 "latent+desc94_switched": pred_ld_sw, "latent+desc94_global": pred_ld_gl,
                 "persistence": occ0.clone()}
    g_rand = torch.Generator().manual_seed(0)
    rand_score = torch.rand(N, generator=g_rand) * 2 - 1
    return preds_img, dict(wr=wr, wc=wc, mass_hat=mass_hat), rand_score


def paired_sem(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return float("nan")
    return float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def score_slate_pool(value_true, value_pred, slate_ids, higher_is_better, k_fixed=K_FIXED,
                      n_resample=N_RESAMPLE, seed=0):
    """Per-slate slateN (K=N exact) + K=fixed resampled reference, for ONE
    model's value_pred. Returns dict with capture list, chosen list, k_fixed
    list (per slate that has >= k_fixed candidates)."""
    caps, chosen, ks = [], [], []
    g = torch.Generator().manual_seed(seed)
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        vt, vp = value_true[rows], value_pred[rows]
        cap = slate_n_capture(vp, vt, higher_is_better)
        caps.append(cap)
        best_idx = int(torch.argmax(vp) if higher_is_better else torch.argmin(vp))
        chosen.append(float(vt[best_idx]))
        n = rows.numel()
        if n >= k_fixed:
            per = []
            for _ in range(n_resample):
                idx = torch.randperm(n, generator=g)[:k_fixed]
                per.append(slate_n_capture(vp[idx], vt[idx], higher_is_better))
            per = [c for c in per if c == c]
            if per:
                ks.append(float(np.mean(per)))
    return dict(capture=caps, chosen=chosen, k_fixed=ks)


def wlt(chosen_a, chosen_b, tol=1e-9):
    w = l = t = 0
    diffs = []
    for a, b, in zip(chosen_a, chosen_b):
        diffs.append(a - b)
        if abs(a - b) <= tol:
            t += 1
        elif a > b:  # 'a' better (both value fns here use "higher achieved value is better"
            w += 1   # after sign-normalising in the caller)
        else:
            l += 1
    return w, l, t, np.array(diffs)


# ---------------------------------------------------------------------------
# Per-(dataset,shape,valuefn) scoring, given predictions + descriptor readout
# ---------------------------------------------------------------------------

MODEL_NAMES = ["visual_switched", "visual_global", "latent+desc94_switched",
               "latent+desc94_global", "desc_pointmass", "persistence", "random"]


def score_cell(occ_true, preds_img, desc_readout, rand_score, slate_ids, shape, value_fn,
               H, W, slate_seed_base):
    # goal mask/field: for random_quadrant, seed must be STABLE PER SLATE
    # (goals.py's own contract) -- since every slate here shares one mask
    # per dataset in this script (we build one mask per (dataset,shape) call,
    # not literally per-slate-seeded-quadrant re-draw), we instead pick ONE
    # random quadrant per DATASET (seed = slate_seed_base), documented below;
    # a true per-slate-seeded quadrant would need a mask per slate, which
    # would fragment the pooled degeneracy stats -- out of scope here, noted
    # as a simplification in RESULTS.md.
    mask, dw = build_goal(shape, H, W, slate_seed=slate_seed_base)

    v_true, v_pred, higher = value_true_and_pred(value_fn, occ_true, preds_img, dw, mask)
    v_pred["desc_pointmass"] = desc_pointmass_value(
        value_fn, desc_readout["wr"], desc_readout["wc"], desc_readout["mass_hat"], dw, mask, H, W)
    v_pred["random"] = rand_score if higher else -rand_score  # random should have no consistent sign pref

    v0 = v_true  # note: v_true already is raw V (V1); "v0" (pre-push) not needed for slateN (see METRICS.md)
    frac_zero = None  # computed by caller from raw dv (occ_true vs occ0) -- see main loop

    out = {}
    for name in MODEL_NAMES:
        stats = score_slate_pool(v_true, v_pred[name], slate_ids, higher_is_better=higher)
        caps = np.array([c for c in stats["capture"] if c == c])
        out[name] = {
            "slateN_mean": float(caps.mean()) if len(caps) else float("nan"),
            "slateN_sem": paired_sem(caps),
            "n_slates": len(caps),
            "slate32_mean": float(np.mean(stats["k_fixed"])) if stats["k_fixed"] else float("nan"),
            "slate32_sem": paired_sem(np.array(stats["k_fixed"])) if len(stats["k_fixed"]) > 1 else float("nan"),
            "_chosen": stats["chosen"],
            "_goodness_chosen": [c if higher else -c for c in stats["chosen"]],
        }
    return out


def head_to_head(out, pairs):
    h2h = {}
    for a, b in pairs:
        w, l, t, diffs = wlt(out[a]["_goodness_chosen"], out[b]["_goodness_chosen"])
        h2h[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "ties": t,
                              "n_slates": len(out[a]["_goodness_chosen"]),
                              "paired_sem_goodness_diff": paired_sem(diffs),
                              "mean_goodness_diff": float(diffs.mean()) if len(diffs) else float("nan")}
    return h2h


def strip_private(out):
    return {name: {k: v for k, v in d.items() if not k.startswith("_")} for name, d in out.items()}


def main():
    t0 = time.time()
    results = {"note": "n20_L10mm EXCLUDED from every pooled number per task instruction.",
               "goal_shapes": GOAL_SHAPES, "value_fns": VALUE_FNS,
               "degeneracy_thresh": DEGENERACY_THRESH, "k_fixed": K_FIXED, "n_resample": N_RESAMPLE}

    print("[1/7] loading persisted operators (no refitting)...")
    cell_v_spec, sw_ops_v, gl_op_v, be_v = load_stage2_visual()
    E, Dec, latent_dim, be_ld, ops_ld, A_ld = load_stage3_latent()
    desc_ops, be_desc, SD = load_desc_operator()

    # -----------------------------------------------------------------
    # Axis 4 verification: the overnight holdout was never trained on.
    # -----------------------------------------------------------------
    print("[2/7] verifying the 43-file holdout split (170/213 train, seed=0, "
          "stratified by spawn mode, descriptors.py::split_files)...")
    files = list_files()
    train_i, test_i = split_files(files, seed=0, holdout_frac=0.2)
    assert len(files) == 213 and len(train_i) == 170 and len(test_i) == 43
    test_cache = torch.load(f"{HYBRID_CACHE}/test_cache.pt", map_location="cpu")
    fid_in_cache = test_cache["file_id"].unique().tolist()
    assert sorted(fid_in_cache) == sorted(test_i), (
        "test_cache.pt's file_id set does not match split_files(seed=0, holdout_frac=0.2)'s "
        "test_i -- cannot claim these files were held out.")
    assert set(fid_in_cache).isdisjoint(set(train_i)), "holdout/train file overlap!"
    train_cache_fids = torch.load(f"{HYBRID_CACHE}/train_cache.pt", map_location="cpu")["file_id"].unique().tolist()
    assert set(fid_in_cache).isdisjoint(set(train_cache_fids)), (
        "test_cache.pt files appear in train_cache.pt (used to fit every operator here) -- "
        "holdout contamination!")
    print(f"  VERIFIED: {len(test_i)} files in test_cache.pt exactly match split_files's test_i, "
          f"zero overlap with train_cache.pt's {len(train_cache_fids)} files "
          f"(train {len(train_i)} + test {len(test_i)} = {len(train_i)+len(test_i)} = {len(files)} total).")
    results["holdout_verification"] = {
        "n_files_total": len(files), "n_train": len(train_i), "n_test": len(test_i),
        "test_cache_file_ids_match_split_test_i": True,
        "train_test_disjoint": True,
        "method": "descriptors.py::list_files()+split_files(seed=0,holdout_frac=0.2) reproduced "
                  "here and compared file-id-set-for-file-id-set against test_cache.pt/"
                  "train_cache.pt's own file_id fields (built by hybrid-vis-desc/build_data.py "
                  "calling the SAME split_files call, seed=0) -- exact match, not a re-derivation "
                  "that could silently diverge.",
    }

    # -----------------------------------------------------------------
    # Build eval cells: 2 slates_multistep datasets (L10mm excluded) + holdout
    # -----------------------------------------------------------------
    all_cells = {}  # dataset_name -> (occ_true(occ1), preds_img, desc_readout, rand_score, slate_ids, H, W, seed)

    print("[3/7] building predict caches: n20_L20mm, n20_L40mm (n20_L10mm excluded)...")
    for ds_name, ds_cfg in MULTISTEP_DATASETS.items():
        slates = load_cell(ds_cfg["eval_cfg"], "train", manifest_path=ds_cfg["manifest"],
                            tag=f"broad_{ds_name}")
        step0 = slates.step_idx == 0

        class _S:
            pass
        s = _S()
        s.occ0 = slates.occ0[step0]
        s.actions = slates.actions[step0]
        s.p_start = slates.p_start[step0]
        s.p_stop = slates.p_stop[step0]
        s.workspace_min = slates.workspace_min
        s.workspace_max = slates.workspace_max

        cache, canon0 = build_cache_multistep(s)
        occ1 = slates.occ1[step0].float()
        slate_ids = slates.slate_idx[step0]
        preds_img, desc_readout, rand_score = predict_all_models(
            cache, canon0, E, Dec, latent_dim, cell_v_spec, sw_ops_v, gl_op_v, be_v,
            be_ld, ops_ld, A_ld, desc_ops, be_desc, SD)
        H, W = occ1.shape[-2:]
        all_cells[ds_name] = dict(occ0=cache["occ0"], occ1=occ1, preds_img=preds_img,
                                   desc_readout=desc_readout, rand_score=rand_score,
                                   slate_ids=slate_ids, H=H, W=W, seed=hash(ds_name) % (2**31))
        print(f"  {ds_name}: {slate_ids.unique().numel()} slates, {int(step0.sum())} step0 rows")

    print("[4/7] building holdout predict cache (43 files, first 128 rows each)...")
    cache_ho, canon0_ho, occ1_ho, slate_ids_ho = build_cache_holdout(test_cache, test_i)
    preds_img_ho, desc_readout_ho, rand_score_ho = predict_all_models(
        cache_ho, canon0_ho, E, Dec, latent_dim, cell_v_spec, sw_ops_v, gl_op_v, be_v,
        be_ld, ops_ld, A_ld, desc_ops, be_desc, SD)
    H_ho, W_ho = occ1_ho.shape[-2:]
    all_cells["overnight_holdout"] = dict(occ0=cache_ho["occ0"], occ1=occ1_ho, preds_img=preds_img_ho,
                                           desc_readout=desc_readout_ho, rand_score=rand_score_ho,
                                           slate_ids=slate_ids_ho, H=H_ho, W=W_ho, seed=999)
    print(f"  overnight_holdout: {slate_ids_ho.unique().numel()} slates, {cache_ho['occ0'].shape[0]} step0 rows")

    # -----------------------------------------------------------------
    # Score every (dataset, shape, value_fn) cell
    # -----------------------------------------------------------------
    print("[5/7] scoring all (dataset, goal-shape, goal-function) cells...")
    results["datasets"] = {}

    for ds_name, cell in all_cells.items():
        results["datasets"][ds_name] = {"n_slates": cell["slate_ids"].unique().numel(), "goals": {}}
        for shape in GOAL_SHAPES:
            results["datasets"][ds_name]["goals"][shape] = {}
            for vfn in VALUE_FNS:
                mask, dw = build_goal(shape, cell["H"], cell["W"], slate_seed=cell["seed"])
                # degeneracy: frac of slates where true V is CONSTANT across the whole slate
                # (the model-agnostic "no candidate makes a difference" signal); reported
                # per (dataset,shape,valuefn) as required.
                v_true_full, _, higher = value_true_and_pred(vfn, cell["occ1"], {}, dw, mask)
                frac_const = 0.0
                sid_u = cell["slate_ids"].unique()
                n_const = 0
                for sid in sid_u.tolist():
                    rows = (cell["slate_ids"] == sid).nonzero(as_tuple=True)[0]
                    vt = v_true_full[rows]
                    if float(vt.max() - vt.min()) < 1e-9:
                        n_const += 1
                frac_const = n_const / max(1, sid_u.numel())
                degenerate = frac_const >= DEGENERACY_THRESH

                cell_out = {"frac_slates_degenerate": frac_const}
                if degenerate:
                    cell_out["excluded"] = True
                    results["datasets"][ds_name]["goals"][shape][vfn] = cell_out
                    continue

                out = score_cell(cell["occ1"], cell["preds_img"], cell["desc_readout"],
                                  cell["rand_score"], cell["slate_ids"], shape, vfn,
                                  cell["H"], cell["W"], cell["seed"])
                pairs = [("visual_switched", "visual_global"),
                         ("latent+desc94_switched", "latent+desc94_global"),
                         ("visual_switched", "persistence"), ("desc_pointmass", "persistence"),
                         ("visual_switched", "random")]
                cell_out["cells"] = strip_private(out)
                cell_out["head_to_head"] = head_to_head(out, pairs)
                results["datasets"][ds_name]["goals"][shape][vfn] = cell_out
        print(f"  {ds_name} done ({time.time()-t0:.1f}s elapsed)")

    # -----------------------------------------------------------------
    # Pooled (n20_L20mm + n20_L40mm ONLY -- L10mm excluded) per (shape,valuefn)
    # -----------------------------------------------------------------
    print("[6/7] pooling n20_L20mm + n20_L40mm (L10mm excluded)...")
    results["pooled_multistep_no_L10mm"] = {}
    for shape in GOAL_SHAPES:
        results["pooled_multistep_no_L10mm"][shape] = {}
        for vfn in VALUE_FNS:
            cells_here = []
            for ds_name in MULTISTEP_DATASETS:
                cell_res = results["datasets"][ds_name]["goals"][shape][vfn]
                if not cell_res.get("excluded"):
                    cells_here.append((ds_name, cell_res))
            if not cells_here:
                results["pooled_multistep_no_L10mm"][shape][vfn] = {"excluded_everywhere": True}
                continue
            # pool by concatenating each dataset's per-slate capture/chosen-goodness lists
            pooled_caps = {n: [] for n in MODEL_NAMES}
            pooled_chosen_good = {n: [] for n in MODEL_NAMES}
            for ds_name, _ in cells_here:
                cell = all_cells[ds_name]
                out = score_cell(cell["occ1"], cell["preds_img"], cell["desc_readout"],
                                  cell["rand_score"], cell["slate_ids"], shape, vfn,
                                  cell["H"], cell["W"], cell["seed"])
                for name in MODEL_NAMES:
                    pooled_chosen_good[name].extend(out[name]["_goodness_chosen"])
                    caps_here = out[name]["slateN_mean"]  # placeholder, replaced below
                # per-slate capture list isn't in `out` (stripped for pooling cleanliness) --
                # recompute once more cheaply via score_slate_pool directly per model:
                mask, dw = build_goal(shape, cell["H"], cell["W"], slate_seed=cell["seed"])
                v_true, v_pred, higher = value_true_and_pred(vfn, cell["occ1"], cell["preds_img"], dw, mask)
                v_pred["desc_pointmass"] = desc_pointmass_value(
                    vfn, cell["desc_readout"]["wr"], cell["desc_readout"]["wc"],
                    cell["desc_readout"]["mass_hat"], dw, mask, cell["H"], cell["W"])
                v_pred["random"] = cell["rand_score"] if higher else -cell["rand_score"]
                for name in MODEL_NAMES:
                    stats = score_slate_pool(v_true, v_pred[name], cell["slate_ids"], higher_is_better=higher,
                                              n_resample=1)  # k_fixed unused here, keep cheap
                    pooled_caps[name].extend([c for c in stats["capture"] if c == c])
            pooled_out = {}
            for name in MODEL_NAMES:
                caps = np.array(pooled_caps[name])
                pooled_out[name] = {"slateN_mean": float(caps.mean()) if len(caps) else float("nan"),
                                     "slateN_sem": paired_sem(caps), "n_slates": len(caps)}
            h2h = {}
            for a, b in [("visual_switched", "visual_global"),
                         ("latent+desc94_switched", "latent+desc94_global"),
                         ("desc_pointmass", "persistence")]:
                w, l, t, diffs = wlt(pooled_chosen_good[a], pooled_chosen_good[b])
                h2h[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "ties": t, "n_slates": len(pooled_chosen_good[a]),
                                      "paired_sem_goodness_diff": paired_sem(diffs),
                                      "mean_goodness_diff": float(diffs.mean()) if len(diffs) else float("nan")}
            results["pooled_multistep_no_L10mm"][shape][vfn] = {"cells": pooled_out, "head_to_head": h2h}

    out_path = f"{OUT_DIR}/results_slaten_broad.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    print(f"[7/7] wrote {out_path}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
