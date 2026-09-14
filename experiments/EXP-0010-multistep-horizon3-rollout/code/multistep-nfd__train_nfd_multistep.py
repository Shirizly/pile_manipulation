"""Fine-tune NFD (UNet, Baselines/NFD/runs/nfd_3ch/unet_best.pth) through a
differentiable 3-step closed-loop rollout: each step's prediction (post-
sigmoid occupancy) is fed back as the next step's occupancy input channel.

Objective: L = lam*L1 + lam^2*L2 + lam^3*L3, Lk = FULL-IMAGE MSE (no swept-
region mask) between predicted and true occupancy at step k. Unmasked is
deliberate: masking is what broke the linear-operator (train_multistep.py)
version's diverging global operator -- the objective was unconstrained
exactly where the value function (lyapunov / mass_in_region, evaluated over
the WHOLE image, not just the swept region) reads.

Reuses (does not reimplement):
  - Baselines/NFD/predictor.py: NFDPredictor (arch + init weights), and its
    3-channel input convention (occ, draw_plate_soft(start) x1.0,
    draw_plate_soft(stop) x1.0).
  - experiments/temp/multistep-rollout/rollout.py: load_dataset (raw
    _{batch}_data.pt files, NOT *_eval configs -- those filter rows and
    break row==env alignment), occ_of, px_of, RawStub, make_nfd_step
    (no-grad inference version, used here only for the untrained baseline
    eval), metrics/swept_region_mask (for reporting per-step accuracy,
    NEVER used in the training loss itself), lyapunov/lyapunov_weights,
    mass_in_region.
  - experiments/temp/multistep-train/train_multistep.py: make_split
    (35/15 slate_idx split, seed 0, same for both datasets), overall script
    shape.

Data: Genesis/data/slates_multistep/{n20_L20mm,n20_L40mm}, n20_L10mm
EXCLUDED per task instruction.

Memory: NFD's UNet is tiny (features=[4,8,16], ~30k params, 64x64 grid), so
a 3-step chained graph is cheap -- no gradient checkpointing needed even at
large batch size on an 8GB card. Batch size actually used is reported in
the printed log and results JSON.

Evaluation (comparable to rollout.py / train_multistep.py):
  - Per-step closed-loop and teacher-forced accuracy (rollout.py's
    `metrics`, swept-region masked, on held-out test rows).
  - Terminal slateN (lyapunov + mass_in_region) at horizon 3: wins/losses/
    ties by the METRICS.md capture-sign definition (capture > 1e-6 = win,
    < -1e-6 = loss, else tie -- NOT an argmax-match definition, which
    train_multistep.py's wins_losses_ties used for a *different* comparison
    (trained-vs-init model agreement) and which the task flagged as reading
    0 everywhere when misapplied here), full-pool (N=128) capture + paired
    sem, and a K=32/reps=50 reference point (METRICS.md's fixed cross-
    dataset reference).
"""
from __future__ import annotations
import sys, os, json, time, copy
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-rollout")

import numpy as np
import torch
import torch.nn.functional as F

from transforms.functional import draw_plate_soft
from fit_linear_foresight import actions_to_pixels, swept_region_mask, metrics
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region
from utils import git_provenance

from rollout import (
    load_dataset, DATASETS, BOUNDS, GRID, PLATE_PX, WS_MIN, WS_MAX,
    occ_of, px_of, RawStub, make_nfd_step,
)
from Baselines.NFD.predictor import NFDPredictor, _plate_geometry_px

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "/home/alon/Code/pile_manipulation"
OUT_DIR = f"{REPO}/experiments/temp/multistep-nfd"
NFD_CKPT = f"{REPO}/Baselines/NFD/runs/nfd_3ch/unet_best.pth"

LAMBDAS = [0.3, 0.5, 0.7, 0.9]
SEED = 0
N_TRAIN_SLATES = 35
N_TEST_SLATES = 15
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4


def make_split(n_slates=50, n_train=35, n_test=15, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_slates)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:n_train + n_test])
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"
    assert len(train_idx) == n_train and len(test_idx) == n_test
    return set(train_idx.tolist()), set(test_idx.tolist())


# =====================================================================
# differentiable NFD step (mirrors rollout.py's make_nfd_step, but keeps
# the graph -- no @torch.no_grad, model left in whatever mode the caller
# set (train() while training, eval() while evaluating))
# =====================================================================

def build_nfd_geometry():
    raw = RawStub()
    plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)
    ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(DEVICE)[:2]
    return raw, plate_x_px, plate_y_px, sigma, ctr_xy


def nfd_step_diff(model, geom, occ_in, p_start_xy, p_stop_xy, angle):
    raw, plate_x_px, plate_y_px, sigma, ctr_xy = geom
    start_px = p_start_xy.to(DEVICE) * raw.to_pxl + ctr_xy
    stop_px = p_stop_xy.to(DEVICE) * raw.to_pxl + ctr_xy
    H, W = occ_in.shape[-2:]
    r_start = draw_plate_soft(start_px, angle.to(DEVICE), (H, W), plate_x_px, plate_y_px,
                               intensity=1.0, sigma=sigma)
    r_stop = draw_plate_soft(stop_px, angle.to(DEVICE), (H, W), plate_x_px, plate_y_px,
                              intensity=1.0, sigma=sigma)
    x = torch.stack([occ_in, r_start, r_stop], dim=1)
    return torch.sigmoid(model(x))[:, 0]


def rollout_3step_diff(model, geom, occ0, A, ang):
    preds = []
    cur = occ0
    for k in range(3):
        cur = nfd_step_diff(model, geom, cur, A[k][:, 0:2], A[k][:, 2:4], ang[:, k])
        preds.append(cur)
    return preds


def full_image_mse(pred, truth):
    return ((pred - truth) ** 2).reshape(pred.shape[0], -1).mean()


# =====================================================================
# evaluation helpers (METRICS.md capture-sign wins/losses/ties, K=32 ref)
# =====================================================================

def slate_capture(vp, vt, higher, slate_ids, uniq_slates):
    if not higher:
        vp = -vp; vt = -vt
    caps = []
    for sl in uniq_slates:
        m = slate_ids == sl
        p, t = vp[m], vt[m]
        bi = int(np.argmax(p))
        chosen = t[bi]; best = t.max(); mean = t.mean()
        denom = best - mean
        if abs(denom) > 1e-9:
            caps.append((chosen - mean) / denom)
    return np.array(caps)


def slate_capture_k(vp, vt, higher, slate_ids, uniq_slates, k=32, reps=50, seed=0):
    if not higher:
        vp = -vp; vt = -vt
    rng = np.random.default_rng(seed)
    caps = []
    for sl in uniq_slates:
        m = slate_ids == sl
        p, t = vp[m], vt[m]
        n = len(p)
        kk = min(k, n)
        for _ in range(reps):
            idx = rng.choice(n, size=kk, replace=False)
            ps, ts = p[idx], t[idx]
            bi = int(np.argmax(ps))
            chosen = ts[bi]; best = ts.max(); mean = ts.mean()
            denom = best - mean
            if abs(denom) > 1e-9:
                caps.append((chosen - mean) / denom)
    return np.array(caps)


def terminal_row(vp, vt, higher, slate_ids, uniq_slates):
    caps = slate_capture(vp, vt, higher, slate_ids, uniq_slates)
    wins = int((caps > 1e-6).sum()); losses = int((caps < -1e-6).sum())
    ties = len(caps) - wins - losses
    caps_k32 = slate_capture_k(vp, vt, higher, slate_ids, uniq_slates, k=32, reps=50)
    return dict(mean_capture=float(caps.mean()),
                sem=float(caps.std(ddof=1) / np.sqrt(len(caps))),
                n_slates=int(len(caps)), wins=wins, losses=losses, ties=ties,
                k32_mean=float(caps_k32.mean()) if len(caps_k32) else float("nan"),
                k32_sem=float(caps_k32.std(ddof=1) / np.sqrt(len(caps_k32))) if len(caps_k32) > 1 else float("nan"))


def eval_model(model, geom, ds, tag, mode):
    """mode: 'teacher_forced' or 'closed_loop'. Returns dict with acc_steps
    and terminal (lyapunov/mass_in_region) rows on the held-out test rows
    of dataset `tag`."""
    d = ds[tag]
    idx = d["is_test"]
    occ0 = d["occ0"][idx]
    occ_true = [d["occ_true"][k][idx] for k in range(4)]
    A = [d["A"][k][idx] for k in range(3)]
    ang = d["ang"][idx]
    s_px = [d["s_px"][k][idx] for k in range(3)]
    e_px = [d["e_px"][k][idx] for k in range(3)]
    slate_id = d["slate_id"][idx]
    uniq = np.unique(slate_id)

    model.eval()
    with torch.no_grad():
        if mode == "teacher_forced":
            preds = []
            for k in range(3):
                preds.append(nfd_step_diff(model, geom, occ_true[k], A[k][:, 0:2], A[k][:, 2:4], ang[:, k]))
        else:
            preds = rollout_3step_diff(model, geom, occ0, A, ang)

    acc_steps = {}
    for st in range(3):
        region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID),
                                    0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
        m = metrics(preds[st], occ_true[st + 1], occ_true[st], region=region)
        acc_steps[st] = float(m["accuracy"])

    dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
    mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
    mask_corner[: GRID // 2, : GRID // 2] = True
    v_true_lyap = lyapunov(occ_true[3], dw_corner).cpu().numpy()
    v_true_mir = mass_in_region(occ_true[3], mask_corner).cpu().numpy()
    v_pred_lyap = lyapunov(preds[2], dw_corner).cpu().numpy()
    v_pred_mir = mass_in_region(preds[2], mask_corner).cpu().numpy()

    terminal = dict(
        lyapunov=terminal_row(v_pred_lyap, v_true_lyap, False, slate_id, uniq),
        mass_in_region=terminal_row(v_pred_mir, v_true_mir, True, slate_id, uniq),
    )
    return dict(acc_steps=acc_steps, terminal=terminal)


# =====================================================================
# main
# =====================================================================

def main():
    t_start = time.time()
    prov = git_provenance()
    print(f"provenance: {prov}")

    train_slates, test_slates = make_split(seed=SEED)
    print(f"split (seed={SEED}): {len(train_slates)} train / {len(test_slates)} test slates")
    print(f"  train: {sorted(train_slates)}")
    print(f"  test:  {sorted(test_slates)}")
    assert train_slates.isdisjoint(test_slates)

    geom = build_nfd_geometry()

    ds = {}
    for tag, root in DATASETS.items():
        data, _ = load_dataset(root)
        N = data["S0"].shape[0]
        n_slates = data["n_slates"]
        occ0 = occ_of(data["S0"]); occ1 = occ_of(data["S1"])
        occ2 = occ_of(data["S2"]); occ3 = occ_of(data["S3"])
        A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
        ang = data["ANG"].to(DEVICE)
        s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
        s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]
        slate_id = data["slate_id"].numpy()
        is_train = np.isin(slate_id, list(train_slates))
        is_test = np.isin(slate_id, list(test_slates))
        print(f"[{tag}] {N} rows ({n_slates} slates x 128 envs); "
              f"{is_train.sum()} train rows / {is_test.sum()} test rows")
        ds[tag] = dict(occ0=occ0, A=A, ang=ang, s_px=s_px, e_px=e_px,
                        slate_id=slate_id, is_train=is_train, is_test=is_test,
                        occ_true=[occ0, occ1, occ2, occ3])

    # combined train tensors across both datasets
    train_occ0 = torch.cat([ds[t]["occ0"][ds[t]["is_train"]] for t in DATASETS])
    train_occ_true = [torch.cat([ds[t]["occ_true"][k][ds[t]["is_train"]] for t in DATASETS])
                       for k in range(4)]
    train_A = [torch.cat([ds[t]["A"][k][ds[t]["is_train"]] for t in DATASETS]) for k in range(3)]
    train_ang = torch.cat([ds[t]["ang"][ds[t]["is_train"]] for t in DATASETS])
    n_train_rows = train_occ0.shape[0]
    print(f"\ncombined train set: {n_train_rows} rows (both datasets, "
          f"{len(train_slates)} slates each)")

    # ---- untrained NFD baseline (for reference / sanity vs task's stated numbers) ----
    base_pred = NFDPredictor(NFD_CKPT, channels=3, name="nfd_untrained")
    base_pred.model.to(DEVICE).eval()
    print("\n=== untrained NFD baseline (closed-loop + teacher-forced) ===")
    baseline_eval = {}
    for tag in DATASETS:
        baseline_eval[tag] = {}
        for mode in ("teacher_forced", "closed_loop"):
            r = eval_model(base_pred.model, geom, ds, tag, mode)
            baseline_eval[tag][mode] = r
            print(f"  [{tag}] {mode}: step1={r['acc_steps'][0]:+.4f} "
                  f"step2={r['acc_steps'][1]:+.4f} step3={r['acc_steps'][2]:+.4f}")

    # ---- fine-tune per lambda ----
    n_iters_per_epoch = (n_train_rows + BATCH_SIZE - 1) // BATCH_SIZE
    total_iters = n_iters_per_epoch * EPOCHS
    print(f"\ntraining config: batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
          f"iters/epoch={n_iters_per_epoch}, total_iters={total_iters}, lr={LR}")

    trained_models = {}
    traces = {}
    rng_np = np.random.default_rng(SEED)

    for lam in LAMBDAS:
        print(f"\n=== training lambda={lam} ===")
        model = NFDPredictor(NFD_CKPT, channels=3, name=f"nfd_lam{lam}").model
        model.to(DEVICE).train()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        trace = []
        it = 0
        t_lam_start = time.time()
        for epoch in range(EPOCHS):
            perm = rng_np.permutation(n_train_rows)
            for bstart in range(0, n_train_rows, BATCH_SIZE):
                idx = perm[bstart:bstart + BATCH_SIZE]
                idx_t = torch.as_tensor(idx, device=DEVICE)
                occ0_b = train_occ0[idx_t]
                occ_true_b = [train_occ_true[k][idx_t] for k in range(4)]
                A_b = [train_A[k][idx_t] for k in range(3)]
                ang_b = train_ang[idx_t]

                opt.zero_grad()
                preds = rollout_3step_diff(model, geom, occ0_b, A_b, ang_b)
                L1 = full_image_mse(preds[0], occ_true_b[1])
                L2 = full_image_mse(preds[1], occ_true_b[2])
                L3 = full_image_mse(preds[2], occ_true_b[3])
                loss = lam * L1 + lam ** 2 * L2 + lam ** 3 * L3
                loss.backward()
                opt.step()
                it += 1
                if it == 1 or it % 50 == 0 or it == total_iters:
                    trace.append((it, float(loss.item()), float(L1.item()),
                                  float(L2.item()), float(L3.item())))
                    print(f"  [lam={lam}] iter {it:4d}/{total_iters}  L={loss.item():.6f} "
                          f"L1={L1.item():.6f} L2={L2.item():.6f} L3={L3.item():.6f}")
                if not np.isfinite(loss.item()):
                    print(f"  [lam={lam}] NON-FINITE LOSS at iter {it}, aborting this lambda")
                    break
            else:
                continue
            break
        print(f"  lambda={lam} trained in {time.time()-t_lam_start:.0f}s ({it} iters)")
        trained_models[lam] = model
        traces[lam] = trace
        torch.save(model.state_dict(), f"{OUT_DIR}/nfd_lam{lam}.pth")
        print(f"  saved {OUT_DIR}/nfd_lam{lam}.pth")

    # ---- evaluate all trained models ----
    print("\n=== evaluating fine-tuned models on held-out slates ===")
    trained_eval = {}
    for lam in LAMBDAS:
        trained_eval[lam] = {}
        for tag in DATASETS:
            trained_eval[lam][tag] = {}
            for mode in ("teacher_forced", "closed_loop"):
                r = eval_model(trained_models[lam], geom, ds, tag, mode)
                trained_eval[lam][tag][mode] = r
                print(f"  [lam={lam}][{tag}] {mode}: step1={r['acc_steps'][0]:+.4f} "
                      f"step2={r['acc_steps'][1]:+.4f} step3={r['acc_steps'][2]:+.4f}")

    results = dict(
        provenance=prov,
        train_slates=sorted(train_slates), test_slates=sorted(test_slates),
        lambdas=LAMBDAS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
        n_train_rows=n_train_rows, total_iters_per_lambda=total_iters,
        baseline_untrained=baseline_eval,
        trained=trained_eval,
        traces=traces,
        wall_time_s=time.time() - t_start,
    )
    with open(f"{OUT_DIR}/results_nfd_multistep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR}/results_nfd_multistep.json")
    print(f"total wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
