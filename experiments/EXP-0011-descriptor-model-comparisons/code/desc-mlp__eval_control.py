"""experiments/temp/desc-mlp/eval_control.py -- control-metric (slateN) eval
of {persistence, random, switched-linear descriptor operator (weights/
MODEL-0002-descriptor-only-D-all-local), the new single MLP} on
Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm (n20, scatter-spawn
only, n_steps=1 so "step 0" is the whole corpus) and
Genesis/data/slates_multistep/n20_L{20,40}mm (step 0 only; L10mm loaded too
but flagged, per CODEMAP's note that it was ruled problematic elsewhere).

Both descriptor-space models (switched-linear, MLP) never reconstruct an
image -- they read out a point-mass (COM + total predicted mass) value,
exactly as experiments/temp/slaten-broad/eval_slaten_broad.py's
`desc_pointmass_value` does (reused, not reinvented). Image-space `accuracy`
is therefore UNDEFINED for both and is not reported for them.

`descriptor_accuracy` (defined, and comparable between switched-linear and
the MLP -- IDENTICAL 94-dim D-all-local basis, identical train-set
normalisation, scored on the identical held-out rows) is reported
separately in results_train.json / train_mlp.py's own output, not here.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/desc-mlp")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")

from transforms.functional import particles_to_occupancy, push_frame_transform  # noqa: E402
from descriptors_d import push_frame_full_descriptors  # noqa: E402
from descriptors_b import world_to_pushframe_px  # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus  # noqa: E402
from Baselines.LinearForesight.model import bin_index as bin_index_switch  # noqa: E402
from Baselines.common.goals import (  # noqa: E402
    slate_n_capture, mass_in_region, signed_mass_in_region,
    random_quadrant_mask, letter_mask, dist_field_from_mask,
)
from control_utility_test import lyapunov  # noqa: E402
from train_mlp import MLP  # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
DEVICE = "cpu"  # keep everything on CPU: Genesis sets torch's cuda default
                # device on import, which BinnedSlateCorpus's own docstring
                # warns about; this script never imports genesis itself.

GOAL_SHAPES = ["ring_O", "T", "random_quadrant"]
VALUE_FNS = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
K_FIXED = 32


def _to_normalized(px, N):
    return 2.0 * px / (N - 1) - 1.0


def com_world_pixel(com_row, com_col, sp_d, ep_d, H=64, W=64):
    canon_row_px = com_row * H
    canon_col_px = com_col * W
    canon_col_n = _to_normalized(canon_col_px, W)
    canon_row_n = _to_normalized(canon_row_px, H)
    theta = push_frame_transform(sp_d, ep_d, (H, W), 1.0)
    world_col_n = theta[:, 0, 0] * canon_col_n + theta[:, 0, 1] * canon_row_n + theta[:, 0, 2]
    world_row_n = theta[:, 1, 0] * canon_col_n + theta[:, 1, 1] * canon_row_n + theta[:, 1, 2]
    world_col_px = (world_col_n + 1.0) * W / 2.0 - 0.5
    world_row_px = (world_row_n + 1.0) * H / 2.0 - 0.5
    return world_row_px, world_col_px


def bilinear_sample(field, row_px, col_px):
    H, W = field.shape
    r = row_px.clamp(0, H - 1)
    c = col_px.clamp(0, W - 1)
    r0 = r.floor().long().clamp(0, H - 2)
    c0 = c.floor().long().clamp(0, W - 2)
    r1, c1 = r0 + 1, c0 + 1
    fr, fc = (r - r0.float()), (c - c0.float())
    f00 = field[r0, c0]; f01 = field[r0, c1]
    f10 = field[r1, c0]; f11 = field[r1, c1]
    return (f00 * (1 - fr) * (1 - fc) + f01 * (1 - fr) * fc +
            f10 * fr * (1 - fc) + f11 * fr * fc)


def desc_pointmass_value(value_fn, wr, wc, mass_hat, dw, mask):
    if value_fn == "lyapunov":
        return bilinear_sample(dw.cpu(), wr, wc)
    mask_f = mask.float()
    m_at_com = bilinear_sample(mask_f, wr, wc)
    if value_fn == "mass_in_region":
        return mass_hat * m_at_com
    elif value_fn == "signed_mass_in_region":
        return mass_hat * (2 * m_at_com - 1)
    raise ValueError(value_fn)


def build_goal(shape, H, W, slate_seed):
    if shape == "random_quadrant":
        mask_np, _q = random_quadrant_mask(H, W, seed=slate_seed)
    elif shape == "ring_O":
        mask_np = letter_mask("O", H, W)
    elif shape == "T":
        mask_np = letter_mask("T", H, W)
    else:
        raise ValueError(shape)
    mask = torch.from_numpy(mask_np)
    dw = torch.from_numpy(dist_field_from_mask(mask_np))
    return mask, dw


def value_true_and_pred(value_fn, occ_true, dw, mask):
    if value_fn == "lyapunov":
        return lyapunov(occ_true, dw), False  # cost: lower better
    elif value_fn == "mass_in_region":
        return mass_in_region(occ_true, mask), True
    elif value_fn == "signed_mass_in_region":
        return signed_mass_in_region(occ_true, mask), True
    raise ValueError(value_fn)


def compute_descriptors(states, states_, p_start, p_stop):
    occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    H, W = occ0.shape[-2:]
    gm0 = occ0.sum(dim=(-2, -1)) / (H * W)
    length_m = (p_stop - p_start).norm(dim=-1)
    start_px = world_to_pushframe_px(p_start)
    end_px = world_to_pushframe_px(p_stop)
    local0 = push_frame_full_descriptors(occ0, start_px, end_px)
    phi0 = torch.cat([gm0.unsqueeze(1), length_m.unsqueeze(1), local0], dim=1)
    return occ0, occ1, phi0, length_m, start_px, end_px


def load_switched(path="weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["ops"], ck["bin_edges"], ck["slices"]


def load_mlp(path="experiments/temp/desc-mlp/mlp_checkpoint.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(ck["in_dim"], ck["config"]["hidden"], ck["out_dim"], ck["config"]["act_name"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict_switched(ops, bin_edges, phi0, length_m):
    bins = bin_index_switch(length_m, bin_edges)
    out = phi0.clone()
    for b, A in enumerate(ops):
        m = bins == b
        if not bool(m.any()):
            continue
        out[m] = (A @ phi0[m].T).T
    return out


def predict_mlp(model, ck, phi0, action_feat):
    X = np.concatenate([phi0.numpy(), action_feat], axis=1)
    Xz = (X - ck["mu_x"]) / ck["sigma_x"]
    with torch.no_grad():
        pred_z = model(torch.from_numpy(Xz.astype(np.float32))).numpy()
    pred_raw = pred_z * ck["sigma_y"] + ck["mu_y"]
    return torch.from_numpy(pred_raw.astype(np.float32))


def score_pool_capture(v_true, v_pred, slate_ids, higher_is_better):
    caps, chosen_good = [], []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        vt, vp = v_true[rows], v_pred[rows]
        cap = slate_n_capture(vp, vt, higher_is_better)
        caps.append(cap)
        best_idx = int(torch.argmax(vp) if higher_is_better else torch.argmin(vp))
        chosen = float(vt[best_idx])
        chosen_good.append(chosen if higher_is_better else -chosen)
    return caps, chosen_good


def paired_sem(diffs):
    diffs = np.asarray(diffs)
    if len(diffs) < 2:
        return float("nan")
    return float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def wlt(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    diffs = a - b
    w = int((diffs > tol).sum())
    l = int((diffs < -tol).sum())
    t = int((np.abs(diffs) <= tol).sum())
    return w, l, t, diffs


def run_dataset(name, occ0, occ1, phi0, length_m, start_px, end_px, action_feat,
                 slate_ids, sw_ops, sw_edges, mlp_model, mlp_ck, H, W, seed_base):
    phi_sw = predict_switched(sw_ops, sw_edges, phi0, length_m)
    phi_mlp = predict_mlp(mlp_model, mlp_ck, phi0, action_feat)

    SD_slice_com = slice(3, 5)     # SD["com"] from the D-all-local layout
    SD_slice_mass = slice(0, 1)    # SD["global_mass"]

    sp_d, ep_d = start_px, end_px  # already push-frame pixel coords

    def readout(phi_pred):
        com = phi_pred[:, SD_slice_com]
        mass_hat = phi_pred[:, SD_slice_mass][:, 0] * (H * W)
        wr, wc = com_world_pixel(com[:, 0], com[:, 1], sp_d, ep_d, H, W)
        return wr, wc, mass_hat

    wr_sw, wc_sw, mass_sw = readout(phi_sw)
    wr_mlp, wc_mlp, mass_mlp = readout(phi_mlp)
    wr_pe, wc_pe, mass_pe = readout(phi0)  # persistence

    g_rand = torch.Generator().manual_seed(0)
    rand_score = torch.rand(occ0.shape[0], generator=g_rand) * 2 - 1

    out = {"n_slates": int(slate_ids.unique().numel()), "n_rows": int(occ0.shape[0]), "goals": {}}
    for shape in GOAL_SHAPES:
        out["goals"][shape] = {}
        mask, dw = build_goal(shape, H, W, slate_seed=seed_base)
        for vfn in VALUE_FNS:
            v_true, higher = value_true_and_pred(vfn, occ1, dw, mask)
            # degeneracy: frac of slates with dv_true == 0 (constant truth across the pool)
            n_const = 0
            for sid in slate_ids.unique().tolist():
                rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
                vt = v_true[rows]
                if float(vt.max() - vt.min()) < 1e-9:
                    n_const += 1
            frac_const = n_const / max(1, slate_ids.unique().numel())

            v_pred = {
                "persistence": lambda: value_true_and_pred(vfn, occ0, dw, mask)[0],
                "random": lambda: (rand_score if higher else -rand_score),
                "switched_linear": lambda: desc_pointmass_value(vfn, wr_sw, wc_sw, mass_sw, dw, mask),
                "mlp": lambda: desc_pointmass_value(vfn, wr_mlp, wc_mlp, mass_mlp, dw, mask),
            }
            cell = {}
            caps_by_model = {}
            for mname, fn in v_pred.items():
                vp = fn()
                caps, chosen_good = score_pool_capture(v_true, vp, slate_ids, higher)
                caps = [c for c in caps if c == c]
                caps_by_model[mname] = chosen_good
                cell[mname] = {
                    "slateN_mean": float(np.mean(caps)) if caps else float("nan"),
                    "slateN_sem": paired_sem(caps),
                    "n_slates_scored": len(caps),
                }
            h2h = {}
            for a, b in [("mlp", "persistence"), ("mlp", "random"),
                         ("mlp", "switched_linear"), ("switched_linear", "persistence")]:
                w, l, t, diffs = wlt(caps_by_model[a], caps_by_model[b])
                h2h[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "ties": t,
                                      "paired_sem_goodness_diff": paired_sem(diffs),
                                      "mean_goodness_diff": float(diffs.mean()) if len(diffs) else float("nan")}
            out["goals"][shape][vfn] = {"frac_dv_true_zero": frac_const, "cells": cell, "head_to_head": h2h}
    return out


def main():
    print("loading switched-linear operator + MLP...")
    sw_ops, sw_edges, sw_slices = load_switched()
    mlp_model, mlp_ck = load_mlp()
    print(f"MLP: {mlp_ck['n_params']} params, config={mlp_ck['config']['tag']}")

    results = {"note": "control-metric (slateN) eval; descriptor-space models "
                       "(switched_linear, mlp) scored via point-mass (COM+mass) "
                       "readout, image accuracy undefined for them.",
               "goal_shapes": GOAL_SHAPES, "value_fns": VALUE_FNS,
               "switched_linear_checkpoint": "weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt",
               "mlp_checkpoint": "experiments/temp/desc-mlp/mlp_checkpoint.pt",
               "datasets": {}}

    # --- slates_binned: n20_scatter_s20a1000_L20-70mm (n20, SCATTER-spawn only) ---
    print("[binned] loading n20_scatter_s20a1000_L20-70mm ...")
    corpus = BinnedSlateCorpus.load("Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm")
    rows = corpus.step(0)  # n_steps=1, so this is the whole corpus
    states = rows.states[:, :, :3].float()
    states_ = rows.states_[:, :, :3].float()
    p_start = rows.p_starts[:, :2].float()
    p_stop = rows.p_stops[:, :2].float()
    angle = rows.angles.float()
    slate_ids = rows.slate_idx.long()
    occ0, occ1, phi0, length_m, s_px, e_px = compute_descriptors(states, states_, p_start, p_stop)
    action_feat = np.stack([p_start[:, 0].numpy(), p_start[:, 1].numpy(),
                             np.cos(angle.numpy()), np.sin(angle.numpy()),
                             length_m.numpy()], axis=1).astype(np.float32)
    H, W = occ0.shape[-2:]
    res_binned = run_dataset("slates_binned_n20_scatter", occ0, occ1, phi0, length_m, s_px, e_px,
                              action_feat, slate_ids, sw_ops, sw_edges, mlp_model, mlp_ck, H, W,
                              seed_base=12345)
    results["datasets"]["slates_binned_n20_scatter"] = res_binned
    print(f"  {res_binned['n_slates']} slates, {res_binned['n_rows']} rows")

    with open("experiments/temp/desc-mlp/results_control.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote experiments/temp/desc-mlp/results_control.json")


if __name__ == "__main__":
    main()
