"""desc-frame-compare: raw-frame vs action-frame analytic descriptors under a
switched-linear (per-push-length-bin) DMDc model.

Frame A (raw): dmdc_baseline.occupancy_descriptors(occ) computed on the
occupancy AS-IS (world frame).
Frame B (action): occ warped into the canonical push frame FIRST
(transforms.functional.to_push_frame + push_frame_validity_mask, same warp
EXP-0004's descriptors_d.py / Baselines/LinearForesight/model.py use), THEN
the IDENTICAL occupancy_descriptors() formula applied to the masked, warped
occupancy. Same descriptor family (mass, COM, central 2nd moments, low-freq
rfft2 block, n_fourier=8, D=87) both sides -- only the frame differs.

Fit: dmdc_baseline.fit_per_action_operators, one operator per push-length
bin (EXP-0003's 6 equal-width-bin scheme over [0, max(train length_m)],
MIN_ROWS_PER_BIN=50 falls back to identity), ridge sweep lam in
{1e-4, 1e-3, 1e-2} (EXP-0004's own sweep) -- same bins/split/sweep for A and B.

Occupancy rasterisation: transforms.functional.particles_to_occupancy,
uniformly across ALL corpora used here (train/test/multistep/binned) --
NOT the registry's CellData/_draw_particle_grid path other baselines use for
slates_multistep. Deliberate: the descriptor operators are fit against
particles_to_occupancy-derived states (matching EXP-0004's own precedent for
overnight_randlen), so evaluating them against a DIFFERENT rasteriser's
"ground truth" would introduce a systematic mismatch between what the
operator sees and what it is scored against. Internally consistent for this
A-vs-B comparison; flagged in RESULTS.md as a possible source of numeric
disagreement with other baselines' reported numbers (which do use the
registry path).

Control metric: point-mass value readout (COM + total predicted mass) fed
into goals.py's mass_in_region / signed_mass_in_region and
control_utility_test.lyapunov, exactly EXP-0004's slaten-broad script
(desc_pointmass_value) generalised to a second frame. Persistence and random
computed from FULL images (no point-mass approximation needed for them).
"""
from __future__ import annotations

import glob
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")

from transforms.functional import (
    particles_to_occupancy, to_push_frame, push_frame_validity_mask,
    push_frame_transform, _to_normalized,
)
from dmdc_baseline import (
    occupancy_descriptors, descriptor_slices, fit_per_action_operators,
)
from control_utility_test import lyapunov
from Baselines.common.goals import (
    slate_n_capture, mass_in_region, signed_mass_in_region,
    random_quadrant_mask, letter_mask, dist_field_from_mask,
)
from Genesis.binned_slate_dataset import BinnedSlateCorpus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/desc-frame-compare"

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
N_FOURIER = 8
MASK_THRESH = 0.5
N_BINS = 6
MIN_ROWS_PER_BIN = 50
RIDGE_LAMS = [1e-4, 1e-3, 1e-2]
K_FIXED = 32
N_RESAMPLE = 150
DEGENERACY_THRESH = 0.90
GOAL_SHAPES = ["random_quadrant", "ring_O", "T"]
VALUE_FNS = ["lyapunov", "mass_in_region", "signed_mass_in_region"]

SLICES = descriptor_slices(N_FOURIER)
D = SLICES["_total"].stop  # 87
print(f"[cfg] D={D}, n_bins={N_BINS}, ridge lams={RIDGE_LAMS}, device={DEVICE}")


# =====================================================================
# 1. Loading raw transitions -> occupancy + actions
# =====================================================================

def occ_from_states(states: torch.Tensor) -> torch.Tensor:
    return particles_to_occupancy(states[:, :, :3], BOUNDS, (GRID, GRID), footprint_radius=RADIUS)


def world_to_pushframe_px(p_xy: torch.Tensor) -> torch.Tensor:
    """World (x,y) metres [B,2] -> push_frame_transform's (col,row) pixel
    pair. Formula copied from EXP-0004's descriptors_b.py (col=y-bin,
    row=x-bin; the axis swap `particles_to_occupancy` vs `push_frame_*`
    requires -- see that file's module docstring)."""
    x, y = p_xy[:, 0], p_xy[:, 1]
    xb = (x - BOUNDS["x_min"]) / (BOUNDS["x_max"] - BOUNDS["x_min"]) * (GRID - 1)
    yb = (y - BOUNDS["y_min"]) / (BOUNDS["y_max"] - BOUNDS["y_min"]) * (GRID - 1)
    return torch.stack([yb, xb], dim=-1)


def load_randlen_dir(root: str):
    # n20 groups only (mixed_n20/piled_n20/scattered_n20) -- piled_n50/
    # scattered_n50 have 50 particles, a different states tensor shape that
    # cannot be pooled with the rest, AND all downstream eval corpora
    # (slates_multistep, slates_binned) are n20 -- keeping the fit corpus at
    # a matching particle count avoids a confound between "frame" and
    # "particle count" when generalising to the eval corpora.
    files = sorted(glob.glob(f"{root}/*n20*/*_data.pt", recursive=True))
    states, states_, p_start, p_stop = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        states.append(d["states"]); states_.append(d["states_"])
        p_start.append(d["p_starts"][:, :2]); p_stop.append(d["p_stops"][:, :2])
    return (torch.cat(states), torch.cat(states_),
            torch.cat(p_start), torch.cat(p_stop), files)


# =====================================================================
# 2. Descriptor computation, both frames
# =====================================================================

def descriptors_raw(occ: torch.Tensor) -> torch.Tensor:
    return occupancy_descriptors(occ, n_fourier=N_FOURIER)


def descriptors_action(occ: torch.Tensor, start_px: torch.Tensor, end_px: torch.Tensor) -> torch.Tensor:
    H, W = occ.shape[-2:]
    canon = to_push_frame(occ, start_px, end_px, out_res=(H, W), scale=1.0)
    mask = push_frame_validity_mask(start_px, end_px, (H, W), scale=1.0)
    valid = (mask >= MASK_THRESH).float()
    return occupancy_descriptors(canon * valid, n_fourier=N_FOURIER)


def com_world_pixel(com_row, com_col, sp_d, ep_d, H=GRID, W=GRID):
    """Predicted canonical(push)-frame COM (normalised /H,/W, occupancy_
    descriptors' own convention) -> WORLD pixel (row,col), via the action's
    own known affine transform. Copied from experiments/temp/stage3-slaten/
    eval_slaten_latent.py::com_world_pixel (cited in that file's own
    docstring), reused verbatim -- pure coordinate algebra, no new formula."""
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
    r = row_px.clamp(0, H - 1); c = col_px.clamp(0, W - 1)
    r0 = r.floor().long().clamp(0, H - 2); c0 = c.floor().long().clamp(0, W - 2)
    r1, c1 = r0 + 1, c0 + 1
    fr, fc = (r - r0.float()), (c - c0.float())
    f00 = field[r0, c0]; f01 = field[r0, c1]; f10 = field[r1, c0]; f11 = field[r1, c1]
    return (f00 * (1 - fr) * (1 - fc) + f01 * (1 - fr) * fc +
            f10 * fr * (1 - fc) + f11 * fr * fc)


# =====================================================================
# 3. Fit / apply (memory-safe, per-bin loop -- avoids dmdc_baseline.
#    apply_operators's known [N,D,D] OOM at large D; harmless here at D=87
#    but kept for hygiene/consistency with EXP-0004's documented workaround)
# =====================================================================

def apply_ops(A: torch.Tensor, phi: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(phi)
    for b in range(A.shape[0]):
        m = bins == b
        if not bool(m.any()):
            continue
        out[m] = (A[b] @ phi[m].T).T
    return out


def bin_index(lengths_m: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(lengths_m, bin_edges[1:-1])


def descriptor_accuracy(phi_pred, phi_true, phi_persist, mu, sigma, drop_slices):
    keep = torch.ones(phi_true.shape[1], dtype=torch.bool)
    for sl in drop_slices:
        keep[sl] = False
    zp = (phi_pred - mu) / sigma
    zt = (phi_true - mu) / sigma
    z0 = (phi_persist - mu) / sigma
    num = (zp[:, keep] - zt[:, keep]).pow(2).mean().sqrt()
    den = (z0[:, keep] - zt[:, keep]).pow(2).mean().sqrt()
    return float(1 - num / den)


# =====================================================================
# main
# =====================================================================

def main():
    t0 = time.time()
    results = {"D": D, "n_bins": N_BINS, "ridge_lams": RIDGE_LAMS,
               "min_rows_per_bin": MIN_ROWS_PER_BIN, "goal_shapes": GOAL_SHAPES,
               "value_fns": VALUE_FNS, "k_fixed": K_FIXED, "n_resample": N_RESAMPLE,
               "degeneracy_thresh": DEGENERACY_THRESH,
               "rasteriser": "transforms.functional.particles_to_occupancy uniformly "
                             "across all corpora (train/test/multistep/binned) -- see module docstring"}

    print("[1/6] loading overnight_randlen_train / _test ...")
    tr_states, tr_states_, tr_ps, tr_pe, tr_files = load_randlen_dir(
        "Genesis/data/overnight_randlen_train")
    te_states, te_states_, te_ps, te_pe, te_files = load_randlen_dir(
        "Genesis/data/overnight_randlen_test")
    print(f"  train files={len(tr_files)} rows={tr_states.shape[0]}  "
          f"test files={len(te_files)} rows={te_states.shape[0]}")

    print("[2/6] rasterising + computing descriptors (both frames, train+test)...")

    CHUNK = 4096  # affine_grid for to_push_frame is the memory hog (~7.6GB GPU); chunk to fit

    def build(states, states_, ps, pe):
        n = states.shape[0]
        length_m_all, phiA_t_all, phiA_t1_all, phiB_t_all, phiB_t1_all = [], [], [], [], []
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            occ0 = occ_from_states(states[sl]).to(DEVICE)
            occ1 = occ_from_states(states_[sl]).to(DEVICE)
            length_m = (pe[sl] - ps[sl]).norm(dim=-1).to(DEVICE)
            start_px = world_to_pushframe_px(ps[sl]).to(DEVICE)
            end_px = world_to_pushframe_px(pe[sl]).to(DEVICE)
            phiA_t_all.append(descriptors_raw(occ0).cpu())
            phiA_t1_all.append(descriptors_raw(occ1).cpu())
            phiB_t_all.append(descriptors_action(occ0, start_px, end_px).cpu())
            phiB_t1_all.append(descriptors_action(occ1, start_px, end_px).cpu())
            length_m_all.append(length_m.cpu())
        return dict(length_m=torch.cat(length_m_all), phiA_t=torch.cat(phiA_t_all),
                    phiA_t1=torch.cat(phiA_t1_all), phiB_t=torch.cat(phiB_t_all),
                    phiB_t1=torch.cat(phiB_t1_all))

    train = build(tr_states, tr_states_, tr_ps, tr_pe)
    test = build(te_states, te_states_, te_ps, te_pe)

    bin_edges = torch.linspace(0.0, float(train["length_m"].max()), N_BINS + 1)
    results["bin_edges_m"] = bin_edges.tolist()
    bins_tr = bin_index(train["length_m"], bin_edges)
    bins_te = bin_index(test["length_m"], bin_edges)
    counts = [int((bins_tr == b).sum()) for b in range(N_BINS)]
    print(f"  bin edges (mm): {[f'{e*1000:.1f}' for e in bin_edges.tolist()]}")
    print(f"  train rows per bin: {counts}")
    results["train_rows_per_bin"] = counts

    # ---- ridge sweep: fit per frame per lam, score descriptor_accuracy on test ----
    print("[3/6] ridge sweep, fitting + scoring descriptor_accuracy on held-out test...")
    drop = [SLICES["const"]]
    fitted = {}  # (frame, lam) -> (A, mu, sigma)
    acc_sweep = {"A": {}, "B": {}}
    for frame, phi_t_tr, phi_t1_tr, phi_t_te, phi_t1_te in [
        ("A", train["phiA_t"], train["phiA_t1"], test["phiA_t"], test["phiA_t1"]),
        ("B", train["phiB_t"], train["phiB_t1"], test["phiB_t"], test["phiB_t1"]),
    ]:
        mu = phi_t1_tr.mean(dim=0)
        sigma = phi_t1_tr.std(dim=0).clamp_min(1e-8)
        for lam in RIDGE_LAMS:
            A_bins, cnt = fit_per_action_operators(phi_t_tr, phi_t1_tr, bins_tr, N_BINS, lam=lam)
            # MIN_ROWS_PER_BIN fallback to identity for thin bins
            for b in range(N_BINS):
                if counts[b] < MIN_ROWS_PER_BIN:
                    A_bins[b] = torch.eye(D)
            pred_te = apply_ops(A_bins, phi_t_te, bins_te)
            acc = descriptor_accuracy(pred_te, phi_t1_te, phi_t_te, mu, sigma, drop)
            acc_sweep[frame][lam] = acc
            fitted[(frame, lam)] = (A_bins, mu, sigma)
            print(f"  frame={frame} lam={lam:g}  descriptor_accuracy(test)={acc:.4f}")
    results["descriptor_accuracy_ridge_sweep"] = acc_sweep

    best_lam = {frame: max(acc_sweep[frame], key=acc_sweep[frame].get) for frame in ("A", "B")}
    results["best_lam"] = best_lam
    print(f"  best lam by descriptor_accuracy: A={best_lam['A']:g} ({acc_sweep['A'][best_lam['A']]:.4f}), "
          f"B={best_lam['B']:g} ({acc_sweep['B'][best_lam['B']]:.4f})")

    # ---- persist every fitted operator with its config ----
    print("[4/6] persisting fitted operators...")
    for (frame, lam), (A_bins, mu, sigma) in fitted.items():
        torch.save(dict(A=A_bins, mu=mu, sigma=sigma, bin_edges=bin_edges,
                         frame=frame, lam=lam, n_fourier=N_FOURIER, slices=SLICES,
                         min_rows_per_bin=MIN_ROWS_PER_BIN, bounds=BOUNDS, grid=GRID,
                         cube_size=CUBE_SIZE, train_rows_per_bin=counts,
                         config="desc-frame-compare/run_experiment.py"),
                   f"{OUT_DIR}/operators/frame{frame}_lam{lam:g}.pt")
    print(f"  wrote {len(fitted)} operator files to {OUT_DIR}/operators/")

    # =====================================================================
    # 5. Control-metric eval corpora: slates_multistep (step0) + slates_binned
    # =====================================================================
    print("[5/6] building eval cells (slates_multistep step0, slates_binned)...")

    def build_eval_cell(states, states_, ps, pe, slate_ids):
        n = states.shape[0]
        occ0_l, occ1_l, len_l, sp_l, ep_l, phiA_l, phiB_l = [], [], [], [], [], [], []
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            occ0 = occ_from_states(states[sl]).to(DEVICE)
            occ1 = occ_from_states(states_[sl]).to(DEVICE)
            length_m = (pe[sl] - ps[sl]).norm(dim=-1).to(DEVICE)
            start_px = world_to_pushframe_px(ps[sl]).to(DEVICE)
            end_px = world_to_pushframe_px(pe[sl]).to(DEVICE)
            phiA_l.append(descriptors_raw(occ0).cpu())
            phiB_l.append(descriptors_action(occ0, start_px, end_px).cpu())
            occ0_l.append(occ0.cpu()); occ1_l.append(occ1.cpu())
            len_l.append(length_m.cpu()); sp_l.append(start_px.cpu()); ep_l.append(end_px.cpu())
        return dict(occ0=torch.cat(occ0_l), occ1=torch.cat(occ1_l), length_m=torch.cat(len_l),
                    start_px=torch.cat(sp_l), end_px=torch.cat(ep_l),
                    phiA_t=torch.cat(phiA_l), phiB_t=torch.cat(phiB_l), slate_ids=slate_ids)

    eval_cells = {}

    # -- slates_multistep, step0 only --
    for ds_name in ["n20_L20mm", "n20_L40mm"]:  # n20_L10mm excluded: codemap flags it
        # "ruled problematic" (single length under-populates 4/6 length bins;
        # EXP-0004's own precedent for this same exclusion).
        root = f"Genesis/data/slates_multistep/{ds_name}"
        manifest = json.loads(open(f"{root}/manifest.json").read())
        step0_batches = [b["batch_idx"] for b in manifest["batches"] if b["step_idx"] == 0]
        slate_of_batch = {b["batch_idx"]: b["slate_idx"] for b in manifest["batches"]}
        states, states_, ps, pe, slate_ids = [], [], [], [], []
        for bidx in step0_batches:
            f = f"{root}/_{bidx}_data.pt"
            d = torch.load(f, map_location="cpu")
            n = d["states"].shape[0]
            states.append(d["states"]); states_.append(d["states_"])
            ps.append(d["p_starts"][:, :2]); pe.append(d["p_stops"][:, :2])
            slate_ids.append(torch.full((n,), slate_of_batch[bidx], dtype=torch.long))
        cell = build_eval_cell(torch.cat(states), torch.cat(states_), torch.cat(ps),
                                torch.cat(pe), torch.cat(slate_ids))
        eval_cells[ds_name] = cell
        print(f"  {ds_name}: {len(step0_batches)} slates, {cell['occ0'].shape[0]} rows")

    # -- slates_binned, n20 scatter (currently the only one) --
    corpus = BinnedSlateCorpus.load("Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm")
    step0 = corpus._steps[0]
    cell = build_eval_cell(step0.states, step0.states_, step0.p_starts[:, :2],
                            step0.p_stops[:, :2], step0.slate_idx.long())
    eval_cells["slates_binned_n20_scatter"] = cell
    print(f"  slates_binned_n20_scatter: {step0.slate_idx.unique().numel()} slates, "
          f"{cell['occ0'].shape[0]} rows (n20, scatter-spawn only -- NOT general)")

    # =====================================================================
    # 6. Score control metric per (corpus, goal shape, value fn), both frames
    #    + persistence + random baselines
    # =====================================================================
    print("[6/6] scoring control metric...")

    def build_goal(shape, H, W, seed):
        if shape == "random_quadrant":
            mask_np, _q = random_quadrant_mask(H, W, seed=seed)
        elif shape in ("ring_O", "T"):
            mask_np = letter_mask("O" if shape == "ring_O" else "T", H, W)
        else:
            raise ValueError(shape)
        mask = torch.from_numpy(mask_np)
        dw = torch.from_numpy(dist_field_from_mask(mask_np)).to(DEVICE)
        return mask, dw

    def true_value(vfn, occ, dw, mask_dev, higher_is_better_out):
        if vfn == "lyapunov":
            higher_is_better_out.append(False)
            return lyapunov(occ.to(DEVICE), dw).cpu()
        elif vfn == "mass_in_region":
            higher_is_better_out.append(True)
            return mass_in_region(occ.to(DEVICE), mask_dev).cpu()
        elif vfn == "signed_mass_in_region":
            higher_is_better_out.append(True)
            return signed_mass_in_region(occ.to(DEVICE), mask_dev).cpu()
        raise ValueError(vfn)

    def pointmass_value(vfn, row_px, col_px, mass_hat, dw, mask_dev):
        row_px, col_px, mass_hat = row_px.to(DEVICE), col_px.to(DEVICE), mass_hat.to(DEVICE)
        if vfn == "lyapunov":
            return bilinear_sample(dw, row_px, col_px).cpu()
        m_at = bilinear_sample(mask_dev.float(), row_px, col_px)
        if vfn == "mass_in_region":
            return (mass_hat * m_at).cpu()
        return (mass_hat * (2 * m_at - 1)).cpu()

    def paired_sem(diffs):
        diffs = np.asarray(diffs)
        return float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")

    def score_slate_pool(v_true, v_pred, slate_ids, higher, k_fixed=K_FIXED, n_resample=N_RESAMPLE, seed=0):
        caps, chosen, ks = [], [], []
        g = torch.Generator().manual_seed(seed)
        for sid in slate_ids.unique().tolist():
            rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
            vt, vp = v_true[rows], v_pred[rows]
            cap = slate_n_capture(vp, vt, higher)
            caps.append(cap)
            best_idx = int(torch.argmax(vp) if higher else torch.argmin(vp))
            chosen.append(float(vt[best_idx]))
            n = rows.numel()
            if n >= k_fixed:
                per = []
                for _ in range(n_resample):
                    idx = torch.randperm(n, generator=g)[:k_fixed]
                    per.append(slate_n_capture(vp[idx], vt[idx], higher))
                per = [c for c in per if c == c]
                if per:
                    ks.append(float(np.mean(per)))
        return dict(capture=caps, chosen=chosen, k_fixed=ks)

    def wlt(chosen_a, chosen_b, tol=1e-9):
        w = l = t = 0
        diffs = []
        for a, b in zip(chosen_a, chosen_b):
            diffs.append(a - b)
            if abs(a - b) <= tol:
                t += 1
            elif a > b:
                w += 1
            else:
                l += 1
        return w, l, t, np.array(diffs)

    A_op, muA, sigA = fitted[("A", best_lam["A"])]
    B_op, muB, sigB = fitted[("B", best_lam["B"])]

    control_results = {}
    for ds_name, cell in eval_cells.items():
        control_results[ds_name] = {"n_slates": cell["slate_ids"].unique().numel(),
                                     "n_rows": cell["occ0"].shape[0], "goals": {}}
        H, W = cell["occ0"].shape[-2:]
        bins_cell = bin_index(cell["length_m"], bin_edges)

        # descriptor-only predictions for A and B (t+1 estimate), point-mass readout
        phiA1_hat = apply_ops(A_op, cell["phiA_t"], bins_cell)
        phiB1_hat = apply_ops(B_op, cell["phiB_t"], bins_cell)

        # frame A: com already in world-normalised [0,1] (row=y,col=x)
        comA = phiA1_hat[:, SLICES["com"]]
        rowA_px = comA[:, 0] * (H - 1)
        colA_px = comA[:, 1] * (W - 1)
        massA_hat = phiA1_hat[:, SLICES["mass"]][:, 0] * (H * W)

        # frame B: com in push-frame-normalised coords -> map to world pixel
        comB = phiB1_hat[:, SLICES["com"]]
        rowB_px, colB_px = com_world_pixel(comB[:, 0].to(DEVICE), comB[:, 1].to(DEVICE),
                                            cell["start_px"].to(DEVICE), cell["end_px"].to(DEVICE), H, W)
        rowB_px, colB_px = rowB_px.cpu(), colB_px.cpu()
        massB_hat = phiB1_hat[:, SLICES["mass"]][:, 0] * (H * W)

        g_rand = torch.Generator().manual_seed(hash(ds_name) % (2 ** 31))
        rand_score = torch.rand(cell["occ0"].shape[0], generator=g_rand) * 2 - 1

        for shape in GOAL_SHAPES:
            control_results[ds_name]["goals"][shape] = {}
            seed = hash((ds_name, shape)) % (2 ** 31)
            mask, dw = build_goal(shape, H, W, seed)
            mask_dev = mask.to(DEVICE)
            for vfn in VALUE_FNS:
                hib = []
                v_true = true_value(vfn, cell["occ1"], dw, mask_dev, hib)
                higher = hib[0]

                sid_u = cell["slate_ids"].unique()
                n_const = 0
                for sid in sid_u.tolist():
                    rows = (cell["slate_ids"] == sid).nonzero(as_tuple=True)[0]
                    vt = v_true[rows]
                    if float(vt.max() - vt.min()) < 1e-9:
                        n_const += 1
                frac_deg = n_const / max(1, sid_u.numel())
                cell_out = {"frac_slates_degenerate": frac_deg}
                if frac_deg >= DEGENERACY_THRESH:
                    cell_out["excluded"] = True
                    control_results[ds_name]["goals"][shape][vfn] = cell_out
                    continue

                v_pred = {
                    "frame_A": pointmass_value(vfn, rowA_px, colA_px, massA_hat, dw, mask_dev),
                    "frame_B": pointmass_value(vfn, rowB_px, colB_px, massB_hat, dw, mask_dev),
                    "persistence": true_value(vfn, cell["occ0"], dw, mask_dev, [])
                        if False else None,  # placeholder, replaced below
                }
                # persistence: FULL image at t (occ0), not a point-mass approx
                if vfn == "lyapunov":
                    v_pred["persistence"] = lyapunov(cell["occ0"].to(DEVICE), dw).cpu()
                elif vfn == "mass_in_region":
                    v_pred["persistence"] = mass_in_region(cell["occ0"].to(DEVICE), mask_dev).cpu()
                else:
                    v_pred["persistence"] = signed_mass_in_region(cell["occ0"].to(DEVICE), mask_dev).cpu()
                v_pred["random"] = rand_score if higher else -rand_score

                out = {}
                for name, vp in v_pred.items():
                    stats = score_slate_pool(v_true, vp, cell["slate_ids"], higher)
                    caps = np.array([c for c in stats["capture"] if c == c])
                    out[name] = {
                        "slateN_mean": float(caps.mean()) if len(caps) else float("nan"),
                        "slateN_sem": paired_sem(caps),
                        "n_slates_eff": len(caps),
                        "slate32_mean": float(np.mean(stats["k_fixed"])) if stats["k_fixed"] else float("nan"),
                        "slate32_sem": paired_sem(stats["k_fixed"]) if len(stats["k_fixed"]) > 1 else float("nan"),
                        "_chosen": stats["chosen"],
                        "_goodness": [c if higher else -c for c in stats["chosen"]],
                    }
                pairs = [("frame_A", "frame_B"), ("frame_A", "persistence"),
                         ("frame_B", "persistence"), ("frame_A", "random"), ("frame_B", "random")]
                h2h = {}
                for a, b in pairs:
                    w, l, t, diffs = wlt(out[a]["_goodness"], out[b]["_goodness"])
                    h2h[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "ties": t,
                                          "n_slates": len(out[a]["_goodness"]),
                                          "paired_sem_goodness_diff": paired_sem(diffs),
                                          "mean_goodness_diff": float(diffs.mean()) if len(diffs) else float("nan")}
                cell_out["cells"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                     for k, v in out.items()}
                cell_out["head_to_head"] = h2h
                control_results[ds_name]["goals"][shape][vfn] = cell_out
        print(f"  {ds_name} done ({time.time()-t0:.1f}s elapsed)")

    results["control_metric"] = control_results

    out_path = f"{OUT_DIR}/results_desc_frame_compare.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"[done] wrote {out_path}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
