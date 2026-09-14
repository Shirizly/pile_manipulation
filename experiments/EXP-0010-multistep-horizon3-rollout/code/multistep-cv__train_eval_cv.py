"""5-fold cross-validation of NFD multistep (lambda=0.7 fixed) fine-tuning,
over ALL 50 slates x 2 datasets (n20_L20mm, n20_L40mm), to resolve RUN-0004's
underpowered +0.048+/-0.034 (~1.4 sigma) control-benefit estimate.

Design: 5 disjoint folds of 10 slates each (seed 0, covers 0..49 exactly
once). Per fold: train on the other 40 slates (both datasets combined,
same as train_nfd_multistep.py), evaluate ONLY on that fold's own 10
held-out slates, closed-loop, terminal (horizon 3). Pooling all 5 folds
gives every one of the 50 slates x 2 datasets = 100 paired observations,
each scored by a model that never saw that slate in training.

Lambda is FIXED at 0.7 (RUN-0003 found the lambda sweep unresolvable
across {0.3,0.5,0.7,0.9}, all within ~0.01 of each other) -- not swept
again here.

Reuses (does not reimplement):
  - experiments/temp/multistep-nfd/train_nfd_multistep.py: build_nfd_geometry,
    nfd_step_diff, rollout_3step_diff, full_image_mse -- same optimiser/
    iteration settings (batch=256, epochs=100, lr=3e-4, Adam), always
    fine-tuned from the same Baselines/NFD/runs/nfd_3ch/unet_best.pth,
    never from scratch.
  - experiments/temp/multistep-rollout/rollout.py: load_dataset (raw
    _{batch}_data.pt files), occ_of, px_of, DATASETS/GRID/PLATE_PX/DEVICE.
  - experiments/temp/multistep-regret/rescore_regret.py: slate_capture,
    slate_capture_k (K=32/reps=50 reference), paired_sem,
    wlt_cross_model_agreement (METRICS.md cross-model-agreement wlt),
    HIGHER_IS_BETTER, VALUE_FNS pattern, value_of.
  - Baselines/NFD/predictor.py: NFDPredictor.
  - Baselines/common/goals.py: mass_in_region, signed_mass_in_region.
  - control_utility_test.py: lyapunov, lyapunov_weights.
  - fit_linear_foresight.py: swept_region_mask, metrics (per-step acc).
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-rollout")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-nfd")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-regret")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")

import numpy as np
import torch

from rollout import load_dataset, DATASETS, GRID, PLATE_PX, DEVICE, REPO, occ_of, px_of
from fit_linear_foresight import swept_region_mask, metrics
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region, signed_mass_in_region
from Baselines.NFD.predictor import NFDPredictor
from utils import git_provenance

from train_nfd_multistep import (
    build_nfd_geometry, nfd_step_diff, rollout_3step_diff, full_image_mse,
)
from rescore_regret import (
    slate_capture, slate_capture_k, paired_sem, wlt_cross_model_agreement,
    HIGHER_IS_BETTER, VALUE_FNS,
)

OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/multistep-cv"
NFD_CKPT = f"{REPO}/Baselines/NFD/runs/nfd_3ch/unet_best.pth"
SEED = 0
N_SLATES = 50
N_FOLDS = 5
LAMBDA = 0.7
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4


def make_folds(n_slates=N_SLATES, n_folds=N_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_slates)
    folds = [np.sort(f).tolist() for f in np.array_split(perm, n_folds)]
    covered = sorted(sum(folds, []))
    assert covered == list(range(n_slates)), "folds do not exactly cover 0..49"
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            assert not (set(folds[i]) & set(folds[j])), f"folds {i},{j} overlap"
    return folds


def value_of(name, occ, dw_corner, mask_corner):
    if name == "lyapunov":
        return lyapunov(occ, dw_corner)
    elif name == "mass_in_region":
        return mass_in_region(occ, mask_corner)
    elif name == "signed_mass_in_region":
        return signed_mass_in_region(occ, mask_corner)
    raise ValueError(name)


def load_all(dw_corner, mask_corner):
    ds = {}
    for tag, root in DATASETS.items():
        data, _ = load_dataset(root)
        n_slates = data["n_slates"]
        assert n_slates == N_SLATES, f"{tag}: expected {N_SLATES} slates, got {n_slates}"
        occ0 = occ_of(data["S0"]); occ1 = occ_of(data["S1"])
        occ2 = occ_of(data["S2"]); occ3 = occ_of(data["S3"])
        A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
        ang = data["ANG"].to(DEVICE)
        s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
        s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]
        slate_id = data["slate_id"].numpy()
        uniq = np.unique(slate_id)
        assert set(uniq.tolist()) == set(range(N_SLATES)), \
            f"{tag}: slate_idx values are not exactly 0..{N_SLATES-1}"
        ds[tag] = dict(occ_true=[occ0, occ1, occ2, occ3], A=A, ang=ang,
                        s_px=s_px, e_px=e_px, slate_id=slate_id, uniq=uniq,
                        N=occ0.shape[0])
    return ds


def closed_loop_forward(model, geom, occ0, A, ang):
    with torch.no_grad():
        model.eval()
        return rollout_3step_diff(model, geom, occ0, A, ang)


def per_step_acc(preds, occ_true, s_px, e_px):
    acc = {}
    for st in range(3):
        region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID),
                                    0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
        m = metrics(preds[st], occ_true[st + 1], occ_true[st], region=region)
        acc[st] = float(m["accuracy"])
    return acc


def main():
    t0 = time.time()
    prov = git_provenance()
    print(f"provenance: {prov}")

    dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
    mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
    mask_corner[: GRID // 2, : GRID // 2] = True

    folds = make_folds()
    print(f"folds (seed={SEED}): {folds}")

    geom = build_nfd_geometry()
    ds = load_all(dw_corner, mask_corner)

    # ---- untrained baseline: single closed-loop forward per dataset, all rows ----
    base_pred = NFDPredictor(NFD_CKPT, channels=3, name="nfd_untrained")
    base_pred.model.to(DEVICE).eval()
    baseline_preds3 = {}   # tag -> occ3_hat (all rows)
    baseline_acc = {}      # tag -> step-> acc (all rows, single global number for reference)
    for tag, d in ds.items():
        preds = closed_loop_forward(base_pred.model, geom, d["occ_true"][0], d["A"], d["ang"])
        baseline_preds3[tag] = preds[2]
        baseline_acc[tag] = per_step_acc(preds, d["occ_true"], d["s_px"], d["e_px"])
        print(f"[baseline][{tag}] all-rows closed-loop acc: {baseline_acc[tag]}")

    # full-size buffers filled fold-by-fold by the model that held that slate out
    trained_preds3 = {tag: torch.zeros_like(baseline_preds3[tag]) for tag in ds}
    # per-row step predictions too (for pooled per-step accuracy), stored per fold and concatenated by mask
    trained_preds_steps = {tag: [torch.zeros_like(baseline_preds3[tag]) for _ in range(3)] for tag in ds}

    fold_traces = []
    for fi, test_slates in enumerate(folds):
        test_set = set(test_slates)
        train_set = set(range(N_SLATES)) - test_set
        print(f"\n=== fold {fi}: test slates {test_slates} ===")

        train_occ0, train_occ_true_k, train_A_k, train_ang = [], [[], [], [], []], [[], [], []], []
        for tag, d in ds.items():
            is_train = np.isin(d["slate_id"], list(train_set))
            idx_t = torch.from_numpy(is_train).to(DEVICE)
            train_occ0.append(d["occ_true"][0][idx_t])
            for k in range(4):
                train_occ_true_k[k].append(d["occ_true"][k][idx_t])
            for k in range(3):
                train_A_k[k].append(d["A"][k][idx_t])
            train_ang.append(d["ang"][idx_t])
        train_occ0 = torch.cat(train_occ0)
        train_occ_true = [torch.cat(train_occ_true_k[k]) for k in range(4)]
        train_A = [torch.cat(train_A_k[k]) for k in range(3)]
        train_ang = torch.cat(train_ang)
        n_train_rows = train_occ0.shape[0]
        print(f"  train rows (both datasets, 40 slates): {n_train_rows}")

        model = NFDPredictor(NFD_CKPT, channels=3, name=f"nfd_fold{fi}_lam{LAMBDA}").model
        model.to(DEVICE).train()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        n_iters_per_epoch = (n_train_rows + BATCH_SIZE - 1) // BATCH_SIZE
        total_iters = n_iters_per_epoch * EPOCHS
        rng_np = np.random.default_rng(SEED + fi)
        it = 0
        t_fold_start = time.time()
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
                loss = LAMBDA * L1 + LAMBDA ** 2 * L2 + LAMBDA ** 3 * L3
                loss.backward()
                opt.step()
                it += 1
                if not np.isfinite(loss.item()):
                    print(f"  fold {fi} NON-FINITE LOSS at iter {it}, aborting")
                    break
            else:
                continue
            break
        print(f"  fold {fi} trained {it} iters in {time.time()-t_fold_start:.0f}s, final loss={loss.item():.6f}")
        fold_traces.append(dict(fold=fi, test_slates=test_slates, iters=it,
                                 final_loss=float(loss.item()), train_time_s=time.time()-t_fold_start))

        torch.save(model.state_dict(), f"{OUT_DIR}/nfd_fold{fi}_lam{LAMBDA}.pth")

        # ---- evaluate ONLY this fold's own held-out slates, per dataset ----
        for tag, d in ds.items():
            is_test = np.isin(d["slate_id"], test_slates)
            idx_t = torch.from_numpy(is_test).to(DEVICE)
            occ0_t = d["occ_true"][0][idx_t]
            occ_true_t = [d["occ_true"][k][idx_t] for k in range(4)]
            A_t = [a[idx_t] for a in d["A"]]
            ang_t = d["ang"][idx_t]
            preds = closed_loop_forward(model, geom, occ0_t, A_t, ang_t)
            trained_preds3[tag][idx_t] = preds[2]
            for k in range(3):
                trained_preds_steps[tag][k][idx_t] = preds[k]
            acc = per_step_acc(preds, occ_true_t, [s[idx_t] for s in d["s_px"]], [e[idx_t] for e in d["e_px"]])
            print(f"  [fold {fi}][{tag}] held-out closed-loop acc: {acc}")

    # ---- pooled per-step accuracy: every row scored by the model that held its slate out ----
    pooled_step_acc = {}
    for st in range(3):
        accs, ns = [], []
        for tag, d in ds.items():
            region = swept_region_mask(d["s_px"][st], d["e_px"][st], (GRID, GRID),
                                        0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
            m = metrics(trained_preds_steps[tag][st], d["occ_true"][st + 1], d["occ_true"][st], region=region)
            accs.append(float(m["accuracy"])); ns.append(d["N"])
        pooled_step_acc[st] = float(np.average(accs, weights=ns))
    print(f"\npooled (CV, both datasets) per-step closed-loop accuracy: {pooled_step_acc}")
    print(f"RUN-0003 reference: step1~+0.099->+0.21x (untrained->trained, single split)")

    # ---- per-value-fn: pooled paired capture diff (n=100 = 5 folds x 2 datasets x 10 slates) ----
    results = dict(provenance=prov, folds=folds, lambda_=LAMBDA,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                    fold_traces=fold_traces, pooled_step_acc=pooled_step_acc,
                    baseline_acc_all_rows=baseline_acc)

    per_vf = {}
    for vf in VALUE_FNS:
        higher = HIGHER_IS_BETTER[vf]
        per_vf[vf] = {"per_dataset": {}, "degeneracy": {}}
        pooled_diff, pooled_wins, pooled_losses, pooled_ties = [], 0, 0, 0
        pooled_caps_trained, pooled_caps_untrained = [], []
        for tag, d in ds.items():
            occ3_true = d["occ_true"][3]
            occ0 = d["occ_true"][0]
            v_true = value_of(vf, occ3_true, dw_corner, mask_corner).cpu().numpy()
            v_start = value_of(vf, occ0, dw_corner, mask_corner).cpu().numpy()
            degeneracy = float(np.mean(np.abs(v_true - v_start) < 1e-9))
            v_pred_trained = value_of(vf, trained_preds3[tag], dw_corner, mask_corner).cpu().numpy()
            v_pred_base = value_of(vf, baseline_preds3[tag], dw_corner, mask_corner).cpu().numpy()
            uniq = d["uniq"]; slate_id = d["slate_id"]
            caps_trained = slate_capture(v_pred_trained, v_true, higher, slate_id, uniq)
            caps_base = slate_capture(v_pred_base, v_true, higher, slate_id, uniq)
            diff = caps_trained - caps_base
            w, l, t = wlt_cross_model_agreement(v_pred_trained, v_pred_base, v_true, higher, slate_id, uniq)
            caps_k_trained = slate_capture_k(v_pred_trained, v_true, higher, slate_id, uniq)
            caps_k_base = slate_capture_k(v_pred_base, v_true, higher, slate_id, uniq)
            per_vf[vf]["per_dataset"][tag] = dict(
                caps_trained_mean=float(caps_trained.mean()), caps_trained_sem=paired_sem(caps_trained),
                caps_untrained_mean=float(caps_base.mean()), caps_untrained_sem=paired_sem(caps_base),
                k32_trained_mean=float(caps_k_trained.mean()) if len(caps_k_trained) else float("nan"),
                k32_untrained_mean=float(caps_k_base.mean()) if len(caps_k_base) else float("nan"),
                paired_diff_mean=float(diff.mean()), paired_diff_sem=paired_sem(diff), n=len(diff),
                wlt_cross_model_agreement=dict(wins=w, losses=l, ties=t),
            )
            per_vf[vf]["degeneracy"][tag] = degeneracy
            pooled_diff.extend(diff.tolist())
            pooled_wins += w; pooled_losses += l; pooled_ties += t
            pooled_caps_trained.extend(caps_trained.tolist())
            pooled_caps_untrained.extend(caps_base.tolist())
        pooled_diff = np.array(pooled_diff)
        mean_d = float(pooled_diff.mean()); sem_d = paired_sem(pooled_diff)
        sigma = mean_d / sem_d if sem_d > 0 else float("nan")
        per_vf[vf]["pooled"] = dict(
            mean=mean_d, sem=sem_d, n=len(pooled_diff), sigma=sigma,
            wins=pooled_wins, losses=pooled_losses, ties=pooled_ties,
            caps_trained_mean=float(np.mean(pooled_caps_trained)),
            caps_untrained_mean=float(np.mean(pooled_caps_untrained)),
        )
        print(f"\n[{vf}] pooled (n={len(pooled_diff)}) paired capture diff: "
              f"{mean_d:+.4f} +/- {sem_d:.4f} ({sigma:+.2f} sigma)  "
              f"wlt(trained vs untrained)={pooled_wins}/{pooled_losses}/{pooled_ties}")

    results["value_fns"] = per_vf
    results["wall_time_s"] = time.time() - t0
    with open(f"{OUT_DIR}/results_cv.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR}/results_cv.json")
    print(f"total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
