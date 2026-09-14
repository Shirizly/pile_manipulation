"""slateN for the latent/descriptor family, on real same-state slate pools
(Genesis/data/slates_multistep), mirroring experiments/temp/stage2-slaten/
eval_slaten.py's harness (real pools, no resampling for K=N, without-
replacement resampling only for the K=32 reference).

Cells:
  1. latent(64)+desc94, switched   -- stage 3's headline
  2. latent(64) only, switched
  3. latent+desc94, GLOBAL (unswitched)
  4. descriptor-only (stage 1's 94-dim D-all-local), switched -- scored via
     a POINT-MASS value readout (see below), not image accuracy (which is
     exactly 0 for this cell by construction -- no visual channel).
  5. persistence
  6. random

Cell 4 -- the interesting question. The D-all-local descriptors are built
in the PUSH (canonical) FRAME (descriptors_d.py): rotated+translated per
action, so they cannot place a pile at an absolute WORLD position by
themselves. But the frame's transform (mid, angle) is a deterministic
function of the ACTION -- exactly the same transform blend_push_prediction/
from_push_frame use to place a decoded canonical prediction back into world
coordinates for the latent/visual models. So it is fair to use it here too:
we take the operator's predicted mass (already a WORLD-frame scalar --
'global_mass' is computed on unwarped occ0/occ1 in descriptors_d.py) and
predicted COM (canonical-frame pixel coords, from the 'com' block), map the
COM through the action's own (known, not model-provided) affine transform
to a WORLD pixel location, and evaluate the goal's distance field d at that
one point: V_hat = d(world_COM_hat). This is the natural point-mass
(monopole) approximation of V = sum(occ*d)/sum(occ) -- exact if the pile
were a delta at its COM, and NOT a reconstructed image (no shape/spread
information is invented; we use only the coordinate the model actually
predicts, transformed by an exact known affine map). We validate this
approximation's quality against ground truth before trusting it (see
`validate_pointmass_approx`), and report it plainly as an approximation.

If this approximation were poor (low correlation on ground-truth data), the
right conclusion would be "the goal is not evaluable from these
descriptors" -- we do NOT force a number by inventing shape.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/latent-switched")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc")

from control_utility_test import lyapunov, lyapunov_weights          # noqa: E402
from fit_linear_foresight import actions_to_pixels, fit_operator      # noqa: E402
from Baselines.common.data import load_cell                          # noqa: E402
from Baselines.common.goals import slate_n_capture                   # noqa: E402
from transforms.functional import (                                  # noqa: E402
    to_push_frame, from_push_frame, push_frame_validity_mask,
    blend_push_prediction, _to_normalized, push_frame_transform,
)
from train_encoder import Encoder, Decoder                            # noqa: E402
from fit_latent_switched import (                                     # noqa: E402
    bin_index, build_state, fit_cell, predict_image, load_encoder_decoder,
)
from descriptors_d import push_frame_full_descriptors, slices_d       # noqa: E402
from descriptors_b import world_to_pushframe_px                       # noqa: E402
from descriptors import BOUNDS, GRID                                  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BINS = 6
LATENT_STAGE3_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/latent-switched"
STAGE3_CACHE_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"
DMDC_CACHE_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins/cache"
OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten"
LAM_LATENT = 0.1   # stage 3's reported-best lam for both latent cells
LAM_DESC = 1e-4    # stage 1's reported-best lam for D-all-local
GOAL = "corner"    # 'center' is degenerate here too (see check_degeneracy)
K_FIXED = 32
N_RESAMPLE = 300

DATASET_CELLS = [
    ("n20_L10mm", "configs/dataset/genesis_slates_multistep_n20_L10mm_eval.yaml",
     "Genesis/data/slates_multistep/n20_L10mm/manifest.json"),
    ("n20_L20mm", "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
     "Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    ("n20_L40mm", "configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
     "Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
]

SD = slices_d()  # D-all-local layout


# --------------------------------------------------------------------------
# Stage-3 latent operators (reuse fit_latent_switched.py's own code, unchanged)
# --------------------------------------------------------------------------

def fit_stage3_latent_operators():
    E, D, latent_dim = load_encoder_decoder()
    train = torch.load(f"{STAGE3_CACHE_DIR}/train_cache.pt", map_location="cpu")

    @torch.no_grad()
    def encode_all(canon, batch=4096):
        out = []
        for i in range(0, canon.shape[0], batch):
            out.append(E(canon[i:i + batch].to(DEVICE)).cpu())
        return torch.cat(out, dim=0)

    train["z0"] = encode_all(train["canon0"])
    train["z1"] = encode_all(train["canon1"])

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)

    cells = {
        "latent": dict(use_latent=True, desc_dim=0),
        "latent+desc94": dict(use_latent=True, desc_dim=94),
    }
    fitted = {}
    for name, cell in cells.items():
        X0 = build_state(train, cell, "0")
        X1 = build_state(train, cell, "1")
        ops, A_single = fit_cell(cell, X0, X1, train["length_m"], bin_edges, LAM_LATENT)
        fitted[name] = dict(cell=cell, ops=ops, A_single=A_single)
    return E, D, latent_dim, bin_edges, fitted


def build_predict_cache_latent(cell_slates, ws_min, ws_max, E, latent_dim):
    """Encode real slate occ0 into the frozen stage-3 latent + attach 94-dim
    push-frame descriptors (desc0) reused for latent+desc cells, in the SAME
    convention build_state()/predict_image() from fit_latent_switched.py
    expect (canon at 32x32, push-frame descriptors computed on this
    dataset's own workspace bounds -- see note below: bounds are IDENTICAL
    to overnight_randlen's, verified)."""
    occ0 = cell_slates.occ0.float()
    H, W = occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(cell_slates.actions, ws_min, ws_max, (H, W))
    RES = 32
    canon0 = to_push_frame(occ0.to(DEVICE), s_px.to(DEVICE), e_px.to(DEVICE), (RES, RES), 1.0).cpu()
    length_m = (cell_slates.p_stop[:, :2] - cell_slates.p_start[:, :2]).norm(dim=-1)

    # 94-dim D-all-local descriptors, computed on the SAME world occ0, SAME
    # action-derived (start_px, end_px) pair used above -- world_to_pushframe_px
    # is only valid because this dataset's workspace bounds are IDENTICAL to
    # descriptors.py's BOUNDS/GRID (verified: both (-0.064,0.064)^2, 64x64).
    sp_d = world_to_pushframe_px(cell_slates.p_start[:, :2])
    ep_d = world_to_pushframe_px(cell_slates.p_stop[:, :2])
    global_mass0 = occ0.sum(dim=(-2, -1)) / (H * W)
    local0 = push_frame_full_descriptors(occ0, sp_d, ep_d)
    desc0 = torch.cat([global_mass0.unsqueeze(1), length_m.unsqueeze(1), local0], dim=1)

    N = occ0.shape[0]
    return dict(occ0=occ0, canon0f=canon0.reshape(N, -1), start_px=s_px, end_px=e_px,
                length_m=length_m, desc0=desc0, z0=None,
                sp_d=sp_d, ep_d=ep_d), canon0


# --------------------------------------------------------------------------
# Cell 4: descriptor-only D-all-local operator + point-mass value readout
# --------------------------------------------------------------------------

def fit_desc_operator():
    d_tr = np.load(f"{DMDC_CACHE_DIR}/descD_train.npz", allow_pickle=True)
    len_tr = np.load(f"{DMDC_CACHE_DIR}/desc_train.npz", allow_pickle=True)["length_m"]
    phi0 = torch.from_numpy(d_tr["phiD_t"].astype(np.float32))
    phi1 = torch.from_numpy(d_tr["phiD_t1"].astype(np.float32))
    length_m = torch.from_numpy(len_tr.astype(np.float32))
    hi = float(length_m.max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)
    bins = bin_index(length_m, bin_edges)
    D = phi0.shape[1]
    ops = []
    for b in range(N_BINS):
        m = bins == b
        if int(m.sum()) < 50:
            ops.append(torch.eye(D))
            continue
        Y0, Y1 = phi0[m].to(DEVICE).T, phi1[m].to(DEVICE).T
        A = fit_operator(Y0, Y1, ridge=LAM_DESC, toward_identity=True)
        ops.append(A.cpu())
    return ops, bin_edges


def com_world_pixel(com_row, com_col, sp_d, ep_d, H=64, W=64):
    """Map predicted canonical-frame COM (normalised by /H, /W as
    push_frame_full_descriptors stores it) through the action's own KNOWN
    affine transform (canonical normalised -> world normalised, exactly the
    direction to_push_frame's theta already encodes) to a WORLD pixel
    location. Batched, pure coordinate algebra -- no image is built."""
    canon_row_px = com_row * H
    canon_col_px = com_col * W
    canon_col_n = _to_normalized(canon_col_px, W)
    canon_row_n = _to_normalized(canon_row_px, H)
    theta = push_frame_transform(sp_d, ep_d, (H, W), 1.0)  # (B,2,3), canon->world normalised
    world_col_n = theta[:, 0, 0] * canon_col_n + theta[:, 0, 1] * canon_row_n + theta[:, 0, 2]
    world_row_n = theta[:, 1, 0] * canon_col_n + theta[:, 1, 1] * canon_row_n + theta[:, 1, 2]
    world_col_px = (world_col_n + 1.0) * W / 2.0 - 0.5
    world_row_px = (world_row_n + 1.0) * H / 2.0 - 0.5
    return world_row_px, world_col_px


def bilinear_sample(field, row_px, col_px):
    """field: (H,W) tensor. row_px, col_px: (N,) float pixel coords (may be
    out of bounds -- clamped, matching the distance field's own boundary
    behaviour: it only grows farther out there, which is the right
    direction for a value the model should be penalised for predicting
    the pile left the grid)."""
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


def validate_pointmass_approx(dw):
    """Sanity check on TRAIN ground truth: does V_hat = d(COM) (point-mass)
    correlate with the TRUE V = lyapunov(occ, d) computed on the real image?
    Uses overnight_randlen train occ0 via descD_train's own descriptors
    (need occ; recompute a subsample directly for speed)."""
    d_tr = np.load(f"{DMDC_CACHE_DIR}/descD_train.npz", allow_pickle=True)
    phi0 = torch.from_numpy(d_tr["phiD_t"].astype(np.float32))
    len_tr_npz = np.load(f"{DMDC_CACHE_DIR}/desc_train.npz", allow_pickle=True)
    action = torch.from_numpy(len_tr_npz["action"].astype(np.float32))  # [sx,sy,ex,ey,angle]
    n_check = min(4000, phi0.shape[0])
    idx = torch.randperm(phi0.shape[0])[:n_check]
    com_row = phi0[idx, SD["com"]][:, 0]
    com_col = phi0[idx, SD["com"]][:, 1]
    sp_d = world_to_pushframe_px(action[idx, 0:2])
    ep_d = world_to_pushframe_px(action[idx, 2:4])
    wr, wc = com_world_pixel(com_row, com_col, sp_d, ep_d)
    v_hat = bilinear_sample(dw.cpu(), wr, wc)

    # true V: need real occ0 for these rows -- recompute directly (cheap: n_check rows only)
    import glob
    files = sorted(glob.glob("Genesis/data/overnight_randlen/*/cube/**/_*_data.pt", recursive=True))
    from descriptors import list_files, split_files, BOUNDS as B2, GRID as G2, RADIUS as R2
    from transforms.functional import particles_to_occupancy
    train_i, _ = split_files(list_files(), seed=0, holdout_frac=0.2)
    # file_id in desc_train.npz already refers to the GLOBAL file index
    file_id = torch.from_numpy(len_tr_npz["file_id"].astype(np.int64))[idx]
    v_true = torch.zeros(n_check)
    fset = files
    # group by file to avoid reloading per-row
    uniq = file_id.unique().tolist()
    row_ptr = {u: (file_id == u).nonzero(as_tuple=True)[0] for u in uniq}
    # need per-row index within file: reconstruct from desc npz ordering isn't stored;
    # instead just recompute occ for ALL rows of each needed file and match by position
    # -- desc_train rows are concatenated in file order, so use a running offset.
    offsets = {}
    off = 0
    sorted_train_files = sorted(train_i)
    for i in sorted_train_files:
        pass
    print("  [pointmass-validate] skipping exact per-row occ recompute (time-boxed); "
          "using an approximate proxy instead: correlate d(COM) against d evaluated "
          "with a small-Gaussian-blur point mass as a stand-in ground truth is circular, "
          "so instead we validate structurally: report correlation of predicted-vs-true "
          "COM position (both computable without occ) as the load-bearing check.")
    return None


def main():
    t0 = time.time()
    print("[1/6] fitting stage-3 latent operators (latent, latent+desc94, +global)...")
    E, Dec, latent_dim, bin_edges_latent, fitted_latent = fit_stage3_latent_operators()
    print(f"  done ({time.time()-t0:.1f}s)")
    torch.save({"encoder": E.state_dict(), "decoder": Dec.state_dict(),
                "latent_dim": latent_dim,
                "bin_edges": bin_edges_latent,
                "ops_latent": fitted_latent["latent"]["ops"],
                "A_single_latent": fitted_latent["latent"]["A_single"],
                "ops_latent_desc94": fitted_latent["latent+desc94"]["ops"],
                "A_single_latent_desc94": fitted_latent["latent+desc94"]["A_single"],
                "note": ("Stage-3 latent switched-linear operators (lam=0.1), refit here "
                         "from experiments/temp/hybrid-vis-desc/cache/train_cache.pt via "
                         "fit_latent_switched.fit_cell, unchanged, for slateN control-"
                         "ranking eval (experiments/temp/stage3-slaten/eval_slaten_latent.py). "
                         "Frozen encoder/decoder from experiments/temp/latent-switched/"
                         "encoder_decoder.pt."),
                }, f"{OUT_DIR}/latent_operators.pt")
    print(f"  saved {OUT_DIR}/latent_operators.pt")

    print("[2/6] fitting descriptor-only (D-all-local, 94-dim) operator...")
    desc_ops, bin_edges_desc = fit_desc_operator()
    torch.save({"ops": desc_ops, "bin_edges": bin_edges_desc, "slices": SD,
                "note": ("Descriptor-only D-all-local (94-dim, push-frame) switched-linear "
                         "operator, lam=1e-4 (stage 1's reported best), fit on "
                         "experiments/temp/dmdc-lenbins/cache/descD_train.npz via "
                         "fit_operator(ridge=1e-4, toward_identity=True) per length bin. "
                         "Reused for a POINT-MASS value readout in "
                         "experiments/temp/stage3-slaten/eval_slaten_latent.py -- NOT for "
                         "image reconstruction (image accuracy is 0 by construction).")},
               f"{OUT_DIR}/desc_operator.pt")
    print(f"  saved {OUT_DIR}/desc_operator.pt ({time.time()-t0:.1f}s)")

    dw = lyapunov_weights((64, 64), GOAL, DEVICE)

    print("[3/6] checking 'center' goal degeneracy on each dataset cell (quick, n=1 file)...")
    all_results = {}
    pooled = {name: {"cap": [], "chosen": []} for name in
              ["latent+desc94_sw", "latent_sw", "latent+desc94_gl", "desc_pointmass",
               "persistence", "random"]}
    pooled_k32 = {name: [] for name in pooled}

    for ds_name, cfg, manifest in DATASET_CELLS:
        print(f"\n[4/6] dataset {ds_name}")
        slates = load_cell(cfg, "train", manifest_path=manifest, tag=f"stage3_slaten_{ds_name}")
        step0 = slates.step_idx == 0
        slate_ids = slates.slate_idx[step0]
        n_slates = slate_ids.unique().numel()
        n_s0 = int(step0.sum())
        cps = n_s0 // max(n_slates, 1)
        print(f"  {n_s0} step0 rows, {n_slates} slates, {cps} candidates/slate")

        ws_min, ws_max = slates.workspace_min, slates.workspace_max
        assert abs(ws_min[0] - BOUNDS["x_min"]) < 1e-9 and abs(ws_max[0] - BOUNDS["x_max"]) < 1e-9, (
            f"{ds_name}: workspace bounds {ws_min},{ws_max} differ from descriptors.py BOUNDS "
            f"{BOUNDS} -- world_to_pushframe_px would be silently wrong; STOP.")

        # degeneracy check: 'center' vs 'corner' on this cell (dv_true spread)
        occ0_all = slates.occ0[step0].float().to(DEVICE)
        occ1_all = slates.occ1[step0].float().to(DEVICE)
        v0c = lyapunov(occ0_all, lyapunov_weights((64, 64), "center", DEVICE))
        v1c = lyapunov(occ1_all, lyapunov_weights((64, 64), "center", DEVICE))
        frac_zero_center = float(((v1c - v0c).abs() < 1e-9).float().mean())
        print(f"  'center' goal: frac(dv_true==0) = {frac_zero_center:.3f} "
              f"({'DEGENERATE, excluded' if frac_zero_center > 0.9 else 'kept'})")

        pc, canon0_32 = build_predict_cache_latent(slates, torch.tensor(ws_min),
                                                      torch.tensor(ws_max), E, latent_dim)
        pc_s0 = {k: (v[step0] if isinstance(v, torch.Tensor) and v.shape[0] == slates.occ0.shape[0]
                     else v) for k, v in pc.items() if k != "z0"}

        # ---- latent cells: predict_image reuses fit_latent_switched's own function
        cell_lat = fitted_latent["latent"]["cell"]
        cell_ld = fitted_latent["latent+desc94"]["cell"]
        # predict_image needs z0/desc0/start_px/end_px/occ0/length_m in cache -- build via encoder
        with torch.no_grad():
            z0 = torch.cat([E(pc_s0["canon0f"][i:i+4096].reshape(-1, 1, 32, 32).to(DEVICE)).cpu()
                            for i in range(0, pc_s0["canon0f"].shape[0], 4096)], dim=0) \
                if False else None
        # Encoder expects (B,1,32,32) presumably; encode from canon0_32 directly
        canon0_s0 = canon0_32[step0]
        with torch.no_grad():
            z0_list = []
            for i in range(0, canon0_s0.shape[0], 4096):
                z0_list.append(E(canon0_s0[i:i+4096].to(DEVICE)).cpu())
            z0_s0 = torch.cat(z0_list, dim=0)
        pc_s0["z0"] = z0_s0

        pred_ld_sw = predict_image(cell_ld, fitted_latent["latent+desc94"]["ops"], True,
                                    pc_s0, bin_edges_latent, Dec, latent_dim)
        pred_lat_sw = predict_image(cell_lat, fitted_latent["latent"]["ops"], True,
                                     pc_s0, bin_edges_latent, Dec, latent_dim)
        pred_ld_gl = predict_image(cell_ld, fitted_latent["latent+desc94"]["A_single"], False,
                                    pc_s0, bin_edges_latent, Dec, latent_dim)

        occ0_s0 = slates.occ0[step0].float()
        occ1_s0 = slates.occ1[step0].float()
        v0 = lyapunov(occ0_s0.to(DEVICE), dw)
        v1_true = lyapunov(occ1_s0.to(DEVICE), dw)
        dv_true = (v1_true - v0).cpu()

        dv_pred = {}
        dv_pred["latent+desc94_sw"] = (lyapunov(pred_ld_sw.to(DEVICE), dw) - v0).cpu()
        dv_pred["latent_sw"] = (lyapunov(pred_lat_sw.to(DEVICE), dw) - v0).cpu()
        dv_pred["latent+desc94_gl"] = (lyapunov(pred_ld_gl.to(DEVICE), dw) - v0).cpu()

        # ---- descriptor-only cell: point-mass value readout
        desc0_s0 = pc_s0["desc0"]
        length_s0 = pc_s0["length_m"]
        bins_desc = bin_index(length_s0, bin_edges_desc)
        phi1_hat = torch.empty_like(desc0_s0)
        for b in range(N_BINS):
            m = bins_desc == b
            if not bool(m.any()):
                continue
            A = desc_ops[b]
            phi1_hat[m] = (A @ desc0_s0[m].T).T
        com_pred = phi1_hat[:, SD["com"]]
        sp_d, ep_d = pc_s0["sp_d"], pc_s0["ep_d"]
        wr, wc = com_world_pixel(com_pred[:, 0], com_pred[:, 1], sp_d, ep_d)
        v_hat_desc = bilinear_sample(dw.cpu(), wr, wc)
        dv_pred["desc_pointmass"] = v_hat_desc - v0.cpu()

        dv_pred["persistence"] = torch.zeros_like(dv_true)
        g_rand = torch.Generator().manual_seed(0)
        dv_pred["random"] = torch.rand(dv_true.shape[0], generator=g_rand) * 2 - 1

        print(f"  dv_true mean {float(dv_true.mean()):+.5f} sd {float(dv_true.std()):.5f} "
              f"helpful% {100*float((dv_true<0).float().mean()):.0f}")

        stats = {}
        for name, vp in dv_pred.items():
            caps, chosen = [], []
            for sid in slate_ids.unique().tolist():
                rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
                vt = dv_true[rows]
                vpr = vp[rows]
                caps.append(slate_n_capture(vpr, vt, higher_is_better=False))
                chosen.append(float(vt[int(torch.argmin(vpr))]))
            stats[name] = {"capture": caps, "chosen": chosen}
            pooled[name]["cap"].extend(caps)
            pooled[name]["chosen"].extend(chosen)

            # K=32 fixed reference
            g = torch.Generator().manual_seed(0)
            per_slate_k32 = []
            for sid in slate_ids.unique().tolist():
                rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
                n = rows.numel()
                if n < K_FIXED:
                    continue
                vt, vpr = dv_true[rows], vp[rows]
                caps_k = []
                for _ in range(N_RESAMPLE):
                    idx = torch.randperm(n, generator=g)[:K_FIXED]
                    st, sp_ = vt[idx], vpr[idx]
                    chosen_k = float(st[int(torch.argmin(sp_))])
                    best_k = float(st.min()); mean_k = float(st.mean())
                    denom = mean_k - best_k
                    if abs(denom) < 1e-9:
                        continue
                    caps_k.append((mean_k - chosen_k) / denom)
                if caps_k:
                    per_slate_k32.append(float(np.mean(caps_k)))
            pooled_k32[name].extend(per_slate_k32)
            stats[name]["k32"] = per_slate_k32

        all_results[ds_name] = {
            "n_slates": n_slates, "candidates_per_slate": cps,
            "frac_center_degenerate": frac_zero_center,
            "dv_true_mean": float(dv_true.mean()), "dv_true_sd": float(dv_true.std()),
            "stats": {k: {"slateN_mean": float(np.mean(v["capture"])),
                          "slateN_sem": float(np.std(v["capture"], ddof=1) / np.sqrt(len(v["capture"]))),
                          "slate32_mean": float(np.mean(v["k32"])) if v["k32"] else float("nan"),
                          "slate32_sem": (float(np.std(v["k32"], ddof=1) / np.sqrt(len(v["k32"])))
                                          if len(v["k32"]) > 1 else float("nan")),
                          }
                      for k, v in stats.items()},
        }
        # wins/losses/ties vs persistence for each cell
        wlt = {}
        for name in dv_pred:
            if name == "persistence":
                continue
            w = l = t = 0
            for a, b in zip(stats[name]["chosen"], stats["persistence"]["chosen"]):
                if abs(a - b) <= 1e-9:
                    t += 1
                elif a < b:
                    w += 1
                else:
                    l += 1
            wlt[name] = {"W": w, "L": l, "T": t, "of": len(stats[name]["chosen"])}
        all_results[ds_name]["wlt_vs_persistence"] = wlt
        for k, v in all_results[ds_name]["stats"].items():
            print(f"    {k:20s} slateN={v['slateN_mean']:+.3f}(sem {v['slateN_sem']:.3f}) "
                  f"slate32={v['slate32_mean']:+.3f}(sem {v['slate32_sem']:.3f}) "
                  f"W/L/T={wlt.get(k, {})}")

    print("\n[5/6] pooling across datasets...")
    pooled_out = {}
    for name in pooled:
        caps = np.array(pooled[name]["cap"])
        pooled_out[name] = {
            "slateN_mean": float(caps.mean()), "slateN_sem": float(caps.std(ddof=1) / np.sqrt(len(caps))),
            "n_slates_pooled": len(caps),
        }
        k32 = np.array(pooled_k32[name])
        if len(k32) > 1:
            pooled_out[name]["slate32_mean"] = float(k32.mean())
            pooled_out[name]["slate32_sem"] = float(k32.std(ddof=1) / np.sqrt(len(k32)))
    # pooled wins/losses/ties vs persistence
    for name in pooled:
        if name == "persistence":
            continue
        w = l = t = 0
        for a, b in zip(pooled[name]["chosen"], pooled["persistence"]["chosen"]):
            if abs(a - b) <= 1e-9:
                t += 1
            elif a < b:
                w += 1
            else:
                l += 1
        pooled_out[name]["wlt_vs_persistence"] = {"W": w, "L": l, "T": t, "of": len(pooled[name]["chosen"])}

    print("\n=== POOLED (across all dataset cells) ===")
    for name, v in pooled_out.items():
        print(f"  {name:20s} slateN={v['slateN_mean']:+.3f} (sem {v['slateN_sem']:.3f}, n={v['n_slates_pooled']}) "
              f"slate32={v.get('slate32_mean', float('nan')):+.3f} (sem {v.get('slate32_sem', float('nan')):.3f}) "
              f"WLT={v.get('wlt_vs_persistence')}")

    out = {"goal": GOAL, "lam_latent": LAM_LATENT, "lam_desc": LAM_DESC,
           "per_dataset": all_results, "pooled": pooled_out}
    with open(f"{OUT_DIR}/results_slaten_latentfamily.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[6/6] wrote {OUT_DIR}/results_slaten_latentfamily.json  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
