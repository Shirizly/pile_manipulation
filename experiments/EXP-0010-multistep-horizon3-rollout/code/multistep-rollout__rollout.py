"""Closed-loop 3-step rollout eval on slates_multistep (n20_L20mm, n20_L40mm).

See task brief in the calling agent's prompt. Key design choices, made under
a hard time budget, documented here rather than silently assumed:

  - Raw .pt files read DIRECTLY (not through PileSweepData/build_dataset),
    because the *_eval configs apply `min_push_length_m` filtering that
    DROPS rows -- that breaks the row-index==env-identity alignment this
    whole closed-loop chain construction depends on. Bypassing the configs
    keeps all 128 envs/slate, all 3 steps, unfiltered.
  - Occupancy rasterisation: `transforms.functional.particles_to_occupancy`,
    footprint_radius convention (RADIUS = 0.5*cube_size/pitch), grid 64x64,
    bounds +-0.064 m -- the same convention used throughout
    experiments/EXP-0004.../descriptors*.py and expB_multistep_eval.py.
  - MODEL-0002 (descriptor-only): TEACHER-FORCED terminal ranking only.
    Closed-loop chaining of the full 94-dim descriptor across steps would
    require transforming shape descriptors (moments2/band_mass/dft) from
    step k's push-frame into step k+1's push-frame -- no such transform
    exists in this repo (world_to_pushframe_px only handles point
    coordinates, i.e. COM, not 2nd-moment/Fourier blocks). Rather than
    inventing an unvalidated one under time pressure, this is scoped out
    and reported as a limitation, not silently assumed solved.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")

import numpy as np
import torch

from transforms.functional import (
    particles_to_occupancy, from_push_frame, push_frame_validity_mask,
    blend_push_prediction, draw_plate_soft,
)
from fit_linear_foresight import actions_to_pixels, canonicalise, swept_region_mask, metrics
from Baselines.LinearForesight.model import bin_index as switched_bin_index
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region
from descriptors_d import push_frame_full_descriptors, slices_d
from descriptors_b import world_to_pushframe_px
from eval_slaten_latent import com_world_pixel, bilinear_sample
from utils import git_provenance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "/home/alon/Code/pile_manipulation"
OUT_DIR = f"{REPO}/experiments/temp/multistep-rollout"

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])
PLATE_PX = 0.04 / 0.128 * GRID  # same convention as expB_multistep_eval.py

DATASETS = {
    "n20_L20mm": f"{REPO}/Genesis/data/slates_multistep/n20_L20mm",
    "n20_L40mm": f"{REPO}/Genesis/data/slates_multistep/n20_L40mm",
}

SD = slices_d()


# =============================================================================
# 1. Data loading -- direct from raw .pt files, no config/dataset-registry
#    path, to keep the full unfiltered 128-envs/slate/step structure.
# =============================================================================

def load_dataset(root):
    manifest = json.loads(open(f"{root}/manifest.json").read())
    by_slate = {}
    for b in manifest["batches"]:
        by_slate.setdefault(b["slate_idx"], {})[b["step_idx"]] = b["batch_idx"]
    slates = sorted(by_slate)
    S0, S1, S2, S3, A0, A1, A2, ANG = [], [], [], [], [], [], [], []
    slate_id = []
    first_pair = None
    for si, sl in enumerate(slates):
        bmap = by_slate[sl]
        assert set(bmap) == {0, 1, 2}, f"slate {sl} missing a step"
        d0 = torch.load(f"{root}/_{bmap[0]}_data.pt", map_location="cpu", weights_only=False)
        d1 = torch.load(f"{root}/_{bmap[1]}_data.pt", map_location="cpu", weights_only=False)
        d2 = torch.load(f"{root}/_{bmap[2]}_data.pt", map_location="cpu", weights_only=False)
        if first_pair is None:
            first_pair = (d0, d1)
        S0.append(d0["states"]); S1.append(d0["states_"])
        S2.append(d1["states_"]); S3.append(d2["states_"])
        A0.append(torch.cat([d0["p_starts"][:, :2], d0["p_stops"][:, :2]], dim=-1))
        A1.append(torch.cat([d1["p_starts"][:, :2], d1["p_stops"][:, :2]], dim=-1))
        A2.append(torch.cat([d2["p_starts"][:, :2], d2["p_stops"][:, :2]], dim=-1))
        ANG.append(torch.stack([d0["angles"], d1["angles"], d2["angles"]], dim=-1))
        slate_id.append(torch.full((d0["states"].shape[0],), sl, dtype=torch.long))
    out = dict(
        S0=torch.cat(S0), S1=torch.cat(S1), S2=torch.cat(S2), S3=torch.cat(S3),
        A0=torch.cat(A0), A1=torch.cat(A1), A2=torch.cat(A2),
        ANG=torch.cat(ANG), slate_id=torch.cat(slate_id),
        n_slates=len(slates),
    )
    return out, first_pair


def verify_trajectory_identity(first_pair):
    d0, d1 = first_pair
    true_chain = d0["states_"]           # predicted end-of-step-0 state
    reported_next = d1["states"]         # step-1 file's own start state
    med = (true_chain - reported_next).norm(dim=-1).median().item()
    perm = torch.randperm(reported_next.shape[0])
    med_shuffled = (true_chain - reported_next[perm]).norm(dim=-1).median().item()
    print(f"[verify] row-identity check (one file pair): median L2 "
          f"aligned={med:.3e}  shuffled-control={med_shuffled:.3e}")
    return med, med_shuffled


# =============================================================================
# 2. Occupancy + action-pixel helpers
# =============================================================================

def occ_of(states):
    return particles_to_occupancy(states[..., :3].to(DEVICE), BOUNDS, (GRID, GRID),
                                   footprint_radius=RADIUS)


def px_of(actions):
    return actions_to_pixels(actions, WS_MIN, WS_MAX, (GRID, GRID))


# =============================================================================
# 3. Model step functions (image space): occ_in, s_px, e_px, length_m -> occ1_hat
# =============================================================================

def make_linear_step(ckpt, desc_dim, switched):
    bin_edges = ckpt["bin_edges"].to(DEVICE)
    ops = [o.to(DEVICE) for o in ckpt["switched_ops"]] if switched else None
    global_op = ckpt["global_op"].to(DEVICE) if not switched else None
    RES = 32

    def step_fn(occ_in, s_px, e_px, length_m):
        canon0 = canonicalise(occ_in, s_px, e_px, RES, 1.0)
        canon0f = canon0.reshape(canon0.shape[0], -1)
        if desc_dim > 0:
            local0 = push_frame_full_descriptors(occ_in, s_px, e_px)
            gm0 = occ_in.sum(dim=(-2, -1)) / (occ_in.shape[-2] * occ_in.shape[-1])
            desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)[:, :desc_dim]
            x0 = torch.cat([canon0f, desc0], dim=1)
        else:
            x0 = canon0f
        if switched:
            bins = switched_bin_index(length_m, bin_edges)
            pred_c = torch.empty(x0.shape[0], RES * RES, device=x0.device)
            for b in range(len(ops)):
                m = bins == b
                pred_c[m] = (ops[b] @ x0[m].T).T[:, :RES * RES]
        else:
            pred_c = (global_op @ x0.T).T[:, :RES * RES]
        canon1 = pred_c.reshape(-1, RES, RES)
        H, W = occ_in.shape[-2:]
        back = from_push_frame(canon1, s_px, e_px, (H, W), 1.0)
        mask = push_frame_validity_mask(s_px, e_px, (H, W), (RES, RES), 1.0)
        return blend_push_prediction(back, occ_in, mask).clamp_(0.0, 1.0)

    return step_fn, bin_edges


class RawStub:
    to_pxl = GRID / (BOUNDS["x_max"] - BOUNDS["x_min"])
    ctr_in_PXL = torch.tensor([GRID / 2, GRID / 2, 0.0])
    resolution_scale = 0.5
    configs = [{"plate": {"size": [0.04, 0.002, 0.01]}}]


def make_nfd_step():
    from Baselines.NFD.predictor import NFDPredictor, _plate_geometry_px
    pred = NFDPredictor(f"{REPO}/Baselines/NFD/runs/nfd_3ch/unet_best.pth", channels=3, name="nfd")
    pred.model.to(DEVICE).eval()
    raw = RawStub()
    plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)
    ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(DEVICE)[:2]

    def step_fn(occ_in, p_start_xy, p_stop_xy, angle):
        start_px = p_start_xy.to(DEVICE) * raw.to_pxl + ctr_xy
        stop_px = p_stop_xy.to(DEVICE) * raw.to_pxl + ctr_xy
        H, W = occ_in.shape[-2:]
        r_start = draw_plate_soft(start_px, angle.to(DEVICE), (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle.to(DEVICE), (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        x = torch.stack([occ_in, r_start, r_stop], dim=1)
        with torch.no_grad():
            return torch.sigmoid(pred.model(x))[:, 0]

    return step_fn


# =============================================================================
# 4. Run one dataset
# =============================================================================

def run_dataset(tag, root):
    print(f"\n=== {tag} ===")
    data, first_pair = load_dataset(root)
    verify_trajectory_identity(first_pair)
    N = data["S0"].shape[0]
    n_slates = data["n_slates"]
    print(f"loaded {N} rows ({n_slates} slates x 128 envs)")

    occ0 = occ_of(data["S0"]); occ1 = occ_of(data["S1"])
    occ2 = occ_of(data["S2"]); occ3 = occ_of(data["S3"])
    A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
    ang = data["ANG"].to(DEVICE)
    length_m = [(A[k][:, 2:4] - A[k][:, 0:2]).norm(dim=-1) for k in range(3)]
    s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
    s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]

    # bin-sequence distribution (switched operator, model0001's bin edges)
    ckpt0001 = torch.load(f"{REPO}/weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                           map_location=DEVICE, weights_only=False)
    bin_edges0001 = ckpt0001["bin_edges"].to(DEVICE)
    bins_per_step = [switched_bin_index(length_m[k], bin_edges0001).cpu().numpy() for k in range(3)]
    seqs = list(zip(bins_per_step[0], bins_per_step[1], bins_per_step[2]))
    from collections import Counter
    seq_counts = Counter(seqs)
    top_seqs = seq_counts.most_common(10)
    print(f"bin-sequence distribution (top 10 of {len(seq_counts)} distinct): {top_seqs}")

    step_fn_sw, _ = make_linear_step(ckpt0001, desc_dim=0, switched=True)
    step_fn_gl, _ = make_linear_step(ckpt0001, desc_dim=0, switched=False)
    ckpt_hy = torch.load(f"{REPO}/experiments/temp/stage2-slaten/operators/hybrid94_lam1.0.pt",
                          map_location=DEVICE, weights_only=False)
    step_fn_hy, _ = make_linear_step(ckpt_hy, desc_dim=94, switched=True)
    nfd_step = make_nfd_step()

    IMG_MODELS = {
        "model0001_switched": lambda o, k, sl: step_fn_sw(o, s_px[k][sl], e_px[k][sl], length_m[k][sl]),
        "model0001_global": lambda o, k, sl: step_fn_gl(o, s_px[k][sl], e_px[k][sl], length_m[k][sl]),
        "hybrid94": lambda o, k, sl: step_fn_hy(o, s_px[k][sl], e_px[k][sl], length_m[k][sl]),
        "nfd": lambda o, k, sl: nfd_step(o, A[k][sl, 0:2], A[k][sl, 2:4], ang[sl, k]),
    }

    occ_true = [occ0, occ1, occ2, occ3]
    BATCH = 512

    def chunked(fn, occ_in, k):
        outs = []
        for i in range(0, occ_in.shape[0], BATCH):
            sl = slice(i, i + BATCH)
            outs.append(fn(occ_in[sl], k, sl))
        return torch.cat(outs, dim=0)

    results_acc = {}  # name -> mode -> step -> acc
    preds_terminal = {}  # name -> mode -> occ3_hat (for terminal ranking)

    for name, fn in IMG_MODELS.items():
        for mode in ("teacher_forced", "closed_loop"):
            t0 = time.time()
            cur_pred = occ0
            preds = []
            for k in range(3):
                inp = occ_true[k] if mode == "teacher_forced" else cur_pred
                cur_pred = chunked(fn, inp, k)
                preds.append(cur_pred)
            acc_steps = {}
            for st in range(3):
                region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID),
                                            0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
                m = metrics(preds[st], occ_true[st + 1], occ_true[st], region=region)
                acc_steps[st] = float(m["accuracy"])
            results_acc.setdefault(name, {})[mode] = acc_steps
            preds_terminal.setdefault(name, {})[mode] = preds[2]
            print(f"  {name:20s} {mode:15s} acc(step1,2,3)="
                  f"{acc_steps[0]:+.4f},{acc_steps[1]:+.4f},{acc_steps[2]:+.4f}  "
                  f"({time.time()-t0:.1f}s)")

    # persistence: nothing moves, closed-loop==teacher-forced==occ0 replicated
    pers_acc = {}
    for st in range(3):
        region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID), 0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
        m = metrics(occ0, occ_true[st + 1], occ_true[st], region=region)
        pers_acc[st] = float(m["accuracy"])
    results_acc["persistence"] = {"teacher_forced": pers_acc, "closed_loop": pers_acc}
    preds_terminal["persistence"] = {"teacher_forced": occ0, "closed_loop": occ0}

    # random: shuffled-row true image at each step (independent per step)
    rand_acc = {}
    for st in range(3):
        perm = torch.randperm(N, device=DEVICE)
        region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID), 0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
        m = metrics(occ_true[st + 1][perm], occ_true[st + 1], occ_true[st], region=region)
        rand_acc[st] = float(m["accuracy"])
    results_acc["random"] = {"teacher_forced": rand_acc, "closed_loop": rand_acc}

    # ================= terminal slateN ranking (horizon 3) =================
    dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
    mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
    mask_corner[: GRID // 2, : GRID // 2] = True

    v_true_lyap = lyapunov(occ3, dw_corner)
    v_true_mir = mass_in_region(occ3, mask_corner)

    slate_ids = data["slate_id"].numpy()
    uniq_slates = np.unique(slate_ids)

    def slate_n_capture_np(vp, vt, higher_is_better, k=None, reps=1, rng=None):
        if not higher_is_better:
            vp = -vp; vt = -vt
        n = len(vp)
        if k is None or k >= n:
            best_idx = int(np.argmax(vp))
            chosen = vt[best_idx]; true_best = vt.max(); mean_true = vt.mean()
            denom = true_best - mean_true
            return [float("nan")] if abs(denom) < 1e-9 else [(chosen - mean_true) / denom], best_idx == int(np.argmax(vt))
        caps = []
        tie = False
        for r in range(reps):
            idx = rng.choice(n, size=k, replace=False)
            vps, vts = vp[idx], vt[idx]
            bi = np.argmax(vps)
            chosen = vts[bi]; true_best = vts.max(); mean_true = vts.mean()
            denom = true_best - mean_true
            if abs(denom) > 1e-9:
                caps.append((chosen - mean_true) / denom)
        return caps, tie

    def terminal_table(value_pred_fn, value_true, higher_is_better, name_list):
        rows = {}
        rng = np.random.default_rng(0)
        for name in name_list:
            vp_full = value_pred_fn(name)
            caps_full, ties_full = [], 0
            caps_k32 = []
            for sl in uniq_slates:
                idx = slate_ids == sl
                vp = vp_full[idx]; vt = value_true.cpu().numpy()[idx]
                c, is_tie = slate_n_capture_np(vp, vt, higher_is_better)
                caps_full.append(c[0])
                ties_full += int(is_tie)
                ck, _ = slate_n_capture_np(vp, vt, higher_is_better, k=32, reps=50, rng=rng)
                caps_k32.extend(ck)
            caps_full = np.array([c for c in caps_full if not np.isnan(c)])
            wins = int((caps_full > 1e-6).sum())
            losses = int((caps_full < -1e-6).sum())
            ties = len(caps_full) - wins - losses
            rows[name] = dict(
                mean_capture=float(caps_full.mean()), sem=float(caps_full.std(ddof=1) / np.sqrt(len(caps_full))),
                n_slates=int(len(caps_full)), wins=wins, losses=losses, ties=ties,
                k32_mean=float(np.mean(caps_k32)) if caps_k32 else float("nan"),
            )
        return rows

    # value_pred per model at horizon-3, teacher-forced and closed-loop
    def make_value_pred_fn(value_fn_name, mode):
        def f(name):
            if name == "oracle":
                occ = occ3
            elif name == "random":
                perm = np.random.default_rng(1).permutation(N)
                occ = occ3[perm]
            elif name == "persistence":
                occ = occ0
            else:
                occ = preds_terminal[name][mode]
            if value_fn_name == "lyapunov":
                return lyapunov(occ, dw_corner).cpu().numpy()
            else:
                return mass_in_region(occ, mask_corner).cpu().numpy()
        return f

    all_names = list(IMG_MODELS) + ["persistence", "random", "oracle"]
    terminal = {}
    for value_fn_name, higher in (("lyapunov", False), ("mass_in_region", True)):
        vt = v_true_lyap if value_fn_name == "lyapunov" else v_true_mir
        for mode in ("teacher_forced", "closed_loop"):
            rows = terminal_table(make_value_pred_fn(value_fn_name, mode), vt, higher, all_names)
            terminal[f"{value_fn_name}__{mode}"] = rows

    # ---- MODEL-0002 descriptor-only: TEACHER-FORCED terminal ranking only ----
    ckpt0002 = torch.load(f"{REPO}/weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt",
                           map_location=DEVICE, weights_only=False)
    ops0002 = [o.to(DEVICE) for o in ckpt0002["ops"]]
    bin_edges0002 = ckpt0002["bin_edges"].to(DEVICE)

    def desc0002_predict(occ_in, s_px_k, e_px_k, length_m_k, p_start_xy, p_stop_xy):
        local0 = push_frame_full_descriptors(occ_in, s_px_k, e_px_k)
        gm0 = occ_in.sum(dim=(-2, -1)) / (occ_in.shape[-2] * occ_in.shape[-1])
        desc0 = torch.cat([gm0[:, None], length_m_k[:, None], local0], dim=1)
        bins = switched_bin_index(length_m_k, bin_edges0002)
        pred = torch.empty_like(desc0)
        for b in range(len(ops0002)):
            m = bins == b
            pred[m] = (ops0002[b] @ desc0[m].T).T
        return pred

    sp_d2 = world_to_pushframe_px(A[2][:, 0:2])
    ep_d2 = world_to_pushframe_px(A[2][:, 2:4])
    pred_desc3 = desc0002_predict(occ2, s_px[2], e_px[2], length_m[2], A[2][:, 0:2], A[2][:, 2:4])
    com_row3 = pred_desc3[:, SD["com"].start]
    com_col3 = pred_desc3[:, SD["com"].start + 1]
    mass3_hat = pred_desc3[:, SD["global_mass"].start] * (GRID * GRID)
    wr3, wc3 = com_world_pixel(com_row3, com_col3, sp_d2, ep_d2, H=GRID, W=GRID)
    lyap_hat3 = bilinear_sample(dw_corner, wr3, wc3)
    mask_f = mask_corner.float()
    mir_hat3 = bilinear_sample(mask_f, wr3, wc3) * mass3_hat

    desc_rows = {}
    for value_fn_name, higher, vhat, vt in (
        ("lyapunov", False, lyap_hat3.cpu().numpy(), v_true_lyap.cpu().numpy()),
        ("mass_in_region", True, mir_hat3.cpu().numpy(), v_true_mir.cpu().numpy()),
    ):
        caps, ties_n = [], 0
        rng = np.random.default_rng(0)
        caps_k32 = []
        for sl in uniq_slates:
            idx = slate_ids == sl
            c, is_tie = slate_n_capture_np(vhat[idx], vt[idx], higher)
            caps.append(c[0]); ties_n += int(is_tie)
            ck, _ = slate_n_capture_np(vhat[idx], vt[idx], higher, k=32, reps=50, rng=rng)
            caps_k32.extend(ck)
        caps = np.array([c for c in caps if not np.isnan(c)])
        wins = int((caps > 1e-6).sum()); losses = int((caps < -1e-6).sum())
        ties_cnt = len(caps) - wins - losses
        desc_rows[value_fn_name] = dict(
            mean_capture=float(caps.mean()), sem=float(caps.std(ddof=1) / np.sqrt(len(caps))),
            n_slates=int(len(caps)), wins=wins, losses=losses,
            ties=ties_cnt, k32_mean=float(np.mean(caps_k32)) if caps_k32 else float("nan"),
        )

    return dict(
        tag=tag, n_rows=N, n_slates=n_slates,
        accuracy=results_acc,
        terminal=terminal,
        model0002_descriptor_terminal_teacher_forced=desc_rows,
        bin_seq_top10=[(list(map(int, s)), c) for s, c in top_seqs],
        n_distinct_bin_seqs=len(seq_counts),
    )


def main():
    prov = git_provenance()
    print(f"provenance: {prov}")
    all_results = {"provenance": prov}
    for tag, root in DATASETS.items():
        all_results[tag] = run_dataset(tag, root)
    with open(f"{OUT_DIR}/results_multistep.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR}/results_multistep.json")


if __name__ == "__main__":
    main()
