"""Multi-step (rollout) gradient training of the linear visual-foresight
operator(s), lambda-weighted closed-loop objective:

    L = lam*L1 + lam^2*L2 + lam^3*L3

where Lk is the masked MSE (swept_region_mask for step k's own action) on
the CLOSED-LOOP rollout (step k's input is the model's own step k-1
prediction; step 1's input is the true occ0). Trains from the closed-form
one-step ridge-toward-identity solution
(weights/MODEL-0001-stage2-visual-switched/checkpoint.pt) as initialisation,
both for the SWITCHED (per-bin) operator set and the GLOBAL (unswitched)
single operator -- the latter is the one the untrained rollout showed
diverging (n20_L20mm CL step3 accuracy -0.104).

Reuses (does not reimplement) the evaluation harness from
experiments/temp/multistep-rollout/rollout.py: load_dataset, occ_of, px_of,
and fit_linear_foresight's canonicalise/swept_region_mask/metrics/
actions_to_pixels, and Baselines.LinearForesight.model.bin_index.

Train/test split: by slate_idx, 35 train / 15 test, SAME split applied to
both n20_L20mm and n20_L40mm (both have slate_idx 0..49), seed 0.
n20_L10mm is excluded per task instruction.

Data is loaded directly from the raw `_{batch}_data.pt` files (not the
`*_eval` configs), inheriting the rollout script's finding that the eval
configs' `min_push_length_m` filtering drops rows and breaks the
row==env-identity alignment this closed-loop chain construction depends on.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-rollout")

import numpy as np
import torch

from transforms.functional import (
    from_push_frame, push_frame_validity_mask, blend_push_prediction,
)
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, swept_region_mask, metrics,
)
from Baselines.LinearForesight.model import bin_index as switched_bin_index
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region
from utils import git_provenance

from rollout import load_dataset, DATASETS, BOUNDS, GRID, PLATE_PX, WS_MIN, WS_MAX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "/home/alon/Code/pile_manipulation"
OUT_DIR = f"{REPO}/experiments/temp/multistep-train"
RES = 32
LAMBDAS = [0.3, 0.5, 0.7, 0.9]
N_ITERS = 150
LR = 2e-3
SEED = 0
N_TRAIN_SLATES = 35
N_TEST_SLATES = 15


def occ_of(states):
    from transforms.functional import particles_to_occupancy
    from rollout import RADIUS
    return particles_to_occupancy(states[..., :3].to(DEVICE), BOUNDS, (GRID, GRID),
                                   footprint_radius=RADIUS)


def px_of(actions):
    return actions_to_pixels(actions, WS_MIN, WS_MAX, (GRID, GRID))


# =====================================================================
# split
# =====================================================================

def make_split(n_slates=50, n_train=35, n_test=15, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_slates)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:n_train + n_test])
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"
    assert len(train_idx) == n_train and len(test_idx) == n_test
    return set(train_idx.tolist()), set(test_idx.tolist())


# =====================================================================
# differentiable per-step operator application (reimplements
# make_linear_step's math but autograd-safe: instead of boolean in-place
# assignment into an uninitialised buffer (fine for inference, but mixes
# poorly with a training loop reusing the same buffer across iterations),
# every bin's candidate is computed for the WHOLE batch and combined with
# a multiplicative mask -- differentiable, and each row still gets exactly
# one bin's operator since the masks partition the batch.
# =====================================================================

def apply_operator(ops, bin_edges, occ_in, s_px, e_px, length_m):
    """ops: list of (D,D) tensors (len 1 == global/unswitched, len>1 ==
    switched, dispatched by bin_edges)."""
    canon0 = canonicalise(occ_in, s_px, e_px, RES, 1.0)
    x0 = canon0.reshape(canon0.shape[0], -1)
    if len(ops) == 1:
        pred_c = (ops[0] @ x0.T).T
    else:
        bins = switched_bin_index(length_m, bin_edges)
        pred_c = torch.zeros_like(x0)
        for b, opb in enumerate(ops):
            cand = (opb @ x0.T).T
            m = (bins == b).unsqueeze(1).to(cand.dtype)
            pred_c = pred_c + cand * m
    canon1 = pred_c.reshape(-1, RES, RES)
    H, W = occ_in.shape[-2:]
    back = from_push_frame(canon1, s_px, e_px, (H, W), 1.0)
    mask = push_frame_validity_mask(s_px, e_px, (H, W), (RES, RES), 1.0)
    return blend_push_prediction(back, occ_in, mask).clamp(0.0, 1.0)


def rollout_3step(ops, bin_edges, occ0, s_px, e_px, length_m):
    """Closed-loop 3-step rollout. Returns list of 3 predictions (grad-on)."""
    preds = []
    cur = occ0
    for k in range(3):
        cur = apply_operator(ops, bin_edges, cur, s_px[k], e_px[k], length_m[k])
        preds.append(cur)
    return preds


def masked_mse(pred, truth, region):
    n = pred.shape[0]
    w = region.reshape(n, -1)
    npix = w.sum(dim=1).clamp_min(1.0)
    d = ((pred - truth) * region).reshape(n, -1)
    return (d.pow(2).sum(dim=1) / npix).mean()


def spectral_stats(op):
    """largest |eigenvalue| (spectral radius) and largest singular value.

    Computed on CPU: GPU `torch.linalg.eigvals` hit a broken nvrtc/cusolver
    complex-abs JIT kernel in this environment (libnvrtc-builtins.so.13.0
    missing) -- CPU LAPACK avoids it and this is a one-off 1024x1024 op, not
    a hot loop, so the cost is negligible.
    """
    op_cpu = op.detach().float().cpu()
    ev = torch.linalg.eigvals(op_cpu)
    sr = ev.abs().max().item()
    sv = torch.linalg.svdvals(op_cpu).max().item()
    return sr, sv


# =====================================================================
# main
# =====================================================================

def main():
    t_start = time.time()
    prov = git_provenance()
    print(f"provenance: {prov}")

    train_slates, test_slates = make_split(seed=SEED)
    print(f"split (seed={SEED}): {len(train_slates)} train slates / "
          f"{len(test_slates)} test slates (same set for both datasets)")
    print(f"  train: {sorted(train_slates)}")
    print(f"  test:  {sorted(test_slates)}")

    # ---- load both datasets, build per-dataset tensors ----
    ds = {}
    for tag, root in DATASETS.items():
        data, _ = load_dataset(root)
        N = data["S0"].shape[0]
        n_slates = data["n_slates"]
        occ0 = occ_of(data["S0"]); occ1 = occ_of(data["S1"])
        occ2 = occ_of(data["S2"]); occ3 = occ_of(data["S3"])
        A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
        length_m = [(A[k][:, 2:4] - A[k][:, 0:2]).norm(dim=-1) for k in range(3)]
        s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
        s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]
        slate_id = data["slate_id"].numpy()
        is_train = np.isin(slate_id, list(train_slates))
        is_test = np.isin(slate_id, list(test_slates))
        print(f"[{tag}] {N} rows ({n_slates} slates x 128 envs); "
              f"{is_train.sum()} train rows / {is_test.sum()} test rows")
        ds[tag] = dict(occ0=occ0, occ1=occ1, occ2=occ2, occ3=occ3, A=A,
                        length_m=length_m, s_px=s_px, e_px=e_px,
                        slate_id=slate_id, is_train=is_train, is_test=is_test,
                        occ_true=[occ0, occ1, occ2, occ3])

    # ---- assemble combined TRAIN tensors across both datasets ----
    def cat_train(key_occ=None, key_list=None):
        pass

    train_occ0 = torch.cat([ds[t]["occ0"][ds[t]["is_train"]] for t in DATASETS])
    train_occ_true = [torch.cat([ds[t]["occ_true"][k][ds[t]["is_train"]] for t in DATASETS])
                       for k in range(4)]
    train_s_px = [torch.cat([ds[t]["s_px"][k][ds[t]["is_train"]] for t in DATASETS])
                  for k in range(3)]
    train_e_px = [torch.cat([ds[t]["e_px"][k][ds[t]["is_train"]] for t in DATASETS])
                  for k in range(3)]
    train_len = [torch.cat([ds[t]["length_m"][k][ds[t]["is_train"]] for t in DATASETS])
                 for k in range(3)]
    n_train_rows = train_occ0.shape[0]
    print(f"\ncombined train set: {n_train_rows} rows "
          f"(both datasets, {len(train_slates)} slates each)")

    train_region = [swept_region_mask(train_s_px[k], train_e_px[k], (GRID, GRID),
                                       0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
                    for k in range(3)]

    # ---- load closed-form initialisation ----
    ckpt0001 = torch.load(f"{REPO}/weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                           map_location=DEVICE, weights_only=False)
    bin_edges = ckpt0001["bin_edges"].to(DEVICE)
    init_switched = [o.to(DEVICE).clone() for o in ckpt0001["switched_ops"]]
    init_global = [ckpt0001["global_op"].to(DEVICE).clone()]
    n_bins = len(init_switched)
    print(f"\ninitialisation: {n_bins}-bin switched operator + 1 global operator, "
          f"each {init_switched[0].shape}, from "
          f"weights/MODEL-0001-stage2-visual-switched/checkpoint.pt")

    def train_ops(ops_init, bin_edges_or_none, lam, tag):
        """ops_init: list of tensors (len 1 or n_bins). Returns trained list
        (new tensors, detached) + loss trace."""
        ops = [o.clone().requires_grad_(True) for o in ops_init]
        edges = bin_edges_or_none if bin_edges_or_none is not None else torch.zeros(2, device=DEVICE)
        opt = torch.optim.Adam(ops, lr=LR)
        trace = []
        for it in range(N_ITERS):
            opt.zero_grad()
            preds = rollout_3step(ops, edges, train_occ0, train_s_px, train_e_px, train_len)
            L1 = masked_mse(preds[0], train_occ_true[1], train_region[0])
            L2 = masked_mse(preds[1], train_occ_true[2], train_region[1])
            L3 = masked_mse(preds[2], train_occ_true[3], train_region[2])
            loss = lam * L1 + lam ** 2 * L2 + lam ** 3 * L3
            loss.backward()
            opt.step()
            if it == 0 or (it + 1) % 50 == 0 or it == N_ITERS - 1:
                trace.append((it + 1, float(loss.item()), float(L1.item()),
                              float(L2.item()), float(L3.item())))
                print(f"  [{tag} lam={lam}] iter {it+1:4d}  L={loss.item():.6f}  "
                      f"L1={L1.item():.6f} L2={L2.item():.6f} L3={L3.item():.6f}")
        return [o.detach().clone() for o in ops], trace

    # ---- init-only loss (lambda-independent numerator terms), for the
    # "loss at initialisation" comparison the task asks for ----
    def eval_loss(ops, edges):
        with torch.no_grad():
            preds = rollout_3step(ops, edges, train_occ0, train_s_px, train_e_px, train_len)
            L1 = masked_mse(preds[0], train_occ_true[1], train_region[0])
            L2 = masked_mse(preds[1], train_occ_true[2], train_region[1])
            L3 = masked_mse(preds[2], train_occ_true[3], train_region[2])
        return float(L1), float(L2), float(L3)

    init_losses_switched = eval_loss(init_switched, bin_edges)
    init_losses_global = eval_loss(init_global, torch.zeros(2, device=DEVICE))
    print(f"\ninit (closed-form) train losses (L1,L2,L3), switched: {init_losses_switched}")
    print(f"init (closed-form) train losses (L1,L2,L3), global:   {init_losses_global}")

    # ---- spectral stats at init ----
    spec_init_switched = [spectral_stats(o) for o in init_switched]
    spec_init_global = [spectral_stats(init_global[0])]
    print(f"\nspectral radius / max singular value at INIT:")
    for b, (sr, sv) in enumerate(spec_init_switched):
        print(f"  switched bin {b}: sr={sr:.4f} sv={sv:.4f}")
    print(f"  global:        sr={spec_init_global[0][0]:.4f} sv={spec_init_global[0][1]:.4f}")

    # ---- train, per lambda, both switched and global ----
    trained = {"switched": {}, "global": {}}
    traces = {"switched": {}, "global": {}}
    for lam in LAMBDAS:
        ops_sw, tr_sw = train_ops(init_switched, bin_edges, lam, "switched")
        trained["switched"][lam] = ops_sw
        traces["switched"][lam] = tr_sw
        ops_gl, tr_gl = train_ops(init_global, None, lam, "global")
        trained["global"][lam] = ops_gl
        traces["global"][lam] = tr_gl

    print(f"\n[{time.time()-t_start:.0f}s elapsed] training done, evaluating on held-out slates ...")

    # =====================================================================
    # Evaluation on held-out test rows, per dataset
    # =====================================================================
    dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
    mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
    mask_corner[: GRID // 2, : GRID // 2] = True

    def eval_on_dataset(tag, ops, edges):
        d = ds[tag]
        idx = d["is_test"]
        occ0 = d["occ0"][idx]
        occ_true = [d["occ_true"][k][idx] for k in range(4)]
        s_px = [d["s_px"][k][idx] for k in range(3)]
        e_px = [d["e_px"][k][idx] for k in range(3)]
        length_m = [d["length_m"][k][idx] for k in range(3)]
        slate_id = d["slate_id"][idx]

        with torch.no_grad():
            preds = rollout_3step(ops, edges, occ0, s_px, e_px, length_m)
        acc_steps = {}
        for st in range(3):
            region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID),
                                        0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
            m = metrics(preds[st], occ_true[st + 1], occ_true[st], region=region)
            acc_steps[st] = float(m["accuracy"])

        v_true_lyap = lyapunov(occ_true[3], dw_corner)
        v_true_mir = mass_in_region(occ_true[3], mask_corner)
        occ3_hat = preds[2]
        v_pred_lyap = lyapunov(occ3_hat, dw_corner).cpu().numpy()
        v_pred_mir = mass_in_region(occ3_hat, mask_corner).cpu().numpy()

        uniq = np.unique(slate_id)

        def slate_capture(vp, vt, higher):
            if not higher:
                vp = -vp; vt = -vt
            caps = []
            for sl in uniq:
                m = slate_id == sl
                p, t = vp[m], vt[m]
                bi = int(np.argmax(p))
                chosen = t[bi]; best = t.max(); mean = t.mean()
                denom = best - mean
                if abs(denom) > 1e-9:
                    caps.append((chosen - mean) / denom)
            caps = np.array(caps)
            return caps

        caps_lyap = slate_capture(v_pred_lyap, v_true_lyap.cpu().numpy(), higher=False)
        caps_mir = slate_capture(v_pred_mir, v_true_mir.cpu().numpy(), higher=True)
        return dict(
            acc_steps=acc_steps,
            lyap_mean=float(caps_lyap.mean()), lyap_sem=float(caps_lyap.std(ddof=1) / np.sqrt(len(caps_lyap))),
            mir_mean=float(caps_mir.mean()), mir_sem=float(caps_mir.std(ddof=1) / np.sqrt(len(caps_mir))),
            n_slates=len(uniq),
            v_pred_lyap=v_pred_lyap, v_pred_mir=v_pred_mir, slate_id=slate_id,
        )

    results = {"provenance": prov, "train_slates": sorted(train_slates),
               "test_slates": sorted(test_slates), "lambdas": LAMBDAS,
               "n_iters": N_ITERS, "lr": LR,
               "init_train_losses": {"switched": init_losses_switched, "global": init_losses_global}}

    # init (closed-form) baseline, evaluated the same way, per dataset
    results["init_eval"] = {}
    for tag in DATASETS:
        results["init_eval"][tag] = {
            "switched": eval_on_dataset(tag, init_switched, bin_edges),
            "global": eval_on_dataset(tag, init_global, torch.zeros(2, device=DEVICE)),
        }

    results["trained_eval"] = {}
    for lam in LAMBDAS:
        results["trained_eval"][lam] = {}
        for kind in ("switched", "global"):
            edges = bin_edges if kind == "switched" else torch.zeros(2, device=DEVICE)
            results["trained_eval"][lam][kind] = {}
            for tag in DATASETS:
                results["trained_eval"][lam][kind][tag] = eval_on_dataset(
                    tag, trained[kind][lam], edges)

    # spectral stats after training
    results["spectral"] = {
        "init": {"switched": spec_init_switched, "global": spec_init_global},
        "trained": {lam: {
            "switched": [spectral_stats(o) for o in trained["switched"][lam]],
            "global": [spectral_stats(trained["global"][lam][0])],
        } for lam in LAMBDAS},
    }

    # wins/losses/ties: cross-model-agreement definition (METRICS.md) --
    # compare the TRAINED model's chosen row against the INIT (closed-form)
    # model's chosen row, per slate. tie = same row chosen by both models
    # (not "matches the true argmax", which the rollout run flagged as
    # reading 0 everywhere).
    def wins_losses_ties(vp_a, vp_b, vt, higher, slate_id, uniq):
        w = l = t = 0
        if not higher:
            vp_a = -vp_a; vp_b = -vp_b
        for sl in uniq:
            m = slate_id == sl
            ia = int(np.argmax(vp_a[m])); ib = int(np.argmax(vp_b[m]))
            if ia == ib:
                t += 1
                continue
            vt_m = vt[m]
            va, vb = vt_m[ia], vt_m[ib]
            if va > vb:
                w += 1
            elif va < vb:
                l += 1
            else:
                t += 1
        return w, l, t

    results["wlt_vs_init"] = {}
    for lam in LAMBDAS:
        results["wlt_vs_init"][lam] = {}
        for kind in ("switched", "global"):
            results["wlt_vs_init"][lam][kind] = {}
            for tag in DATASETS:
                init_e = results["init_eval"][tag][kind]
                tr_e = results["trained_eval"][lam][kind][tag]
                d = ds[tag]
                idx = d["is_test"]
                slate_id = d["slate_id"][idx]
                uniq = np.unique(slate_id)
                v_true_lyap = lyapunov(d["occ_true"][3][idx], dw_corner).cpu().numpy()
                v_true_mir = mass_in_region(d["occ_true"][3][idx], mask_corner).cpu().numpy()
                w_l, l_l, t_l = wins_losses_ties(tr_e["v_pred_lyap"], init_e["v_pred_lyap"],
                                                  v_true_lyap, False, slate_id, uniq)
                w_m, l_m, t_m = wins_losses_ties(tr_e["v_pred_mir"], init_e["v_pred_mir"],
                                                  v_true_mir, True, slate_id, uniq)
                results["wlt_vs_init"][lam][kind][tag] = {
                    "lyapunov": {"wins": w_l, "losses": l_l, "ties": t_l, "n": len(uniq)},
                    "mass_in_region": {"wins": w_m, "losses": l_m, "ties": t_m, "n": len(uniq)},
                }

    # strip large arrays before JSON dump
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in
                    ("v_pred_lyap", "v_pred_mir", "slate_id")}
        return d

    results_json = strip(results)
    with open(f"{OUT_DIR}/results_multistep_train.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR}/results_multistep_train.json")

    # ---- persist fitted operators ----
    for lam in LAMBDAS:
        torch.save({
            "switched_ops": [o.cpu() for o in trained["switched"][lam]],
            "bin_edges": bin_edges.cpu(),
            "lam": lam,
            "note": (f"Multi-step (3-step closed-loop rollout) trained SWITCHED operator, "
                     f"lambda={lam}, L=lam*L1+lam^2*L2+lam^3*L3 (masked MSE per step, "
                     f"swept_region_mask per that step's own action), {N_ITERS} Adam "
                     f"iters lr={LR}, initialised from "
                     f"weights/MODEL-0001-stage2-visual-switched/checkpoint.pt "
                     f"(closed-form ridge-toward-identity). Trained on 70 slates "
                     f"(35 n20_L20mm + 35 n20_L40mm, slate_idx seed={SEED} split), "
                     f"held out 15+15 test slates. Provenance: {prov}"),
        }, f"{OUT_DIR}/operator_switched_lam{lam}.pt")
        torch.save({
            "global_op": trained["global"][lam][0].cpu(),
            "lam": lam,
            "note": (f"Multi-step (3-step closed-loop rollout) trained GLOBAL "
                     f"(unswitched) operator, lambda={lam}, same objective/split/init "
                     f"as operator_switched_lam{lam}.pt (initialised from "
                     f"MODEL-0001's global_op). This is the operator family that "
                     f"DIVERGES untrained under closed-loop rollout "
                     f"(n20_L20mm CL step3 accuracy -0.104); trained here to test "
                     f"whether rollout training fixes that. Provenance: {prov}"),
        }, f"{OUT_DIR}/operator_global_lam{lam}.pt")
    print(f"saved 4 switched + 4 global operator .pt files to {OUT_DIR}")
    print(f"\ntotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
