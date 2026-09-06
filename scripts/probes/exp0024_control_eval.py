"""EXP-0024: does the UNet's fine-detail image-accuracy advantage over the
linear operator (EXP-0021/EXP-0022) buy anything for control?

Trains/evaluates both model classes on Genesis/data/cube_spectrum/n20 (the
exact material/action configuration -- 20 piled cubes, heap spawn, 5mm,
fixed 20mm contact-aware pushes -- that Genesis/data/slates/n20_heap_5mm
(EXP-0012) was collected under), then scores BOTH on:

  (a) the standard image metric `accuracy` (fit_linear_foresight.py::metrics),
      on held-out cube_spectrum/n20 transitions (same split for both models);
  (b) the standard control metric `slate4` / `spearman`
      (control_utility_test.py::rank_metrics), on the n20_heap_5mm same-state
      slates (50 states x ~32 candidates, EXP-0012), reusing
      scripts/probes/same_state_degradation.py's per_slate_metrics unchanged.

Reference models: persistence (cannot rank -- dv_pred=0 always, reported but
flagged degenerate), mean-delta, and the oracle (dv_pred := dv_true, the
ranking ceiling; image accuracy is trivially 1.0 so not a meaningful anchor
there and is reported with that caveat).

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0024_control_eval.py \
        configs/dataset/genesis_cube_spectrum_n20.yaml \
        configs/dataset/genesis_cube_spectrum_n20_slates.yaml \
        runs_exp0024/unetfilm_cube_spectrum_n20 --tag cube_spectrum_n20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask,
)
from registry.dataset_registry import build_dataset
from scripts.probes.exp0009_rerun import predict_meandelta
from scripts.probes.exp0021_eval import load_unet, unet_forward
from scripts.probes.same_state_degradation import per_slate_metrics
from control_utility_test import lyapunov, lyapunov_weights

R, CR, RIDGE = 64, 1.0, 1.0  # matches EXP-0008/EXP-0012's fit for this exact
                              # dataset/slate pairing (NOT EXP-0021's headline
                              # crop=0.5 cell -- that choice was for the
                              # granularity datasets' ~40mm pushes).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_cfg")
    ap.add_argument("slate_cfg")
    ap.add_argument("run_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-slate", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== EXP-0024 control eval: {args.tag} ===")

    # ---- train split: fit linear baselines (identity-ridge operator + mean-delta) ----
    data_tr = load_transition_arrays(args.dataset_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    print(f"train: {data_tr.occ_t.shape[0]} transitions, grid {H}x{W}")

    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR).reshape(data_tr.occ_t.shape[0], -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR).reshape(data_tr.occ_t.shape[0], -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)

    model, run_cfg, missing = load_unet(args.run_dir)
    print(f"  UNet load: {missing}")

    # =====================================================================
    # (a) IMAGE ACCURACY on held-out cube_spectrum/n20 (same split, same
    #     code path, for both models)
    # =====================================================================
    cfg = yaml.safe_load(open(args.dataset_cfg).read())
    wrapper_te = build_dataset(cfg, "test")
    n_te = len(wrapper_te)
    raw_te = wrapper_te.raw_dataset
    occ_t = torch.stack([raw_te[i][0][0][0] for i in range(n_te)])
    occ_t1 = torch.stack([raw_te[i][1] for i in range(n_te)])
    actions_te = torch.stack([raw_te.get_raw_action(i) for i in range(n_te)])
    s_te, e_te = actions_to_pixels(actions_te, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region_te = swept_region_mask(s_te, e_te, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    print(f"test (image accuracy): {n_te} transitions")

    X_te = torch.stack([wrapper_te[i]["input"] for i in range(n_te)])
    P_te = torch.stack([wrapper_te[i]["physics"] for i in range(n_te)])
    unet_pred_te = unet_forward(model, X_te, P_te)

    img_preds = {
        "persistence": occ_t,
        "mean-delta": predict_meandelta(bmd, occ_t, s_te, e_te, R, (H, W), CR),
        "linear (ridge->identity)": predict_world(A, occ_t, s_te, e_te, R, (H, W), CR),
        "UNet": unet_pred_te,
        "oracle": occ_t1,
    }
    accuracy = {}
    for name, pred in img_preds.items():
        m = metrics(pred, occ_t1, occ_t, region=region_te)
        accuracy[name] = m["accuracy"]
    print("\n-- image accuracy (held-out cube_spectrum/n20, swept region) --")
    for name, acc in accuracy.items():
        print(f"  {name:26s} accuracy={acc:+.4f}")

    # =====================================================================
    # (b) CONTROL UTILITY on the n20_heap_5mm same-state slates
    # =====================================================================
    scfg = yaml.safe_load(open(args.slate_cfg).read())
    wrapper_sl = build_dataset(scfg, "test")
    raw_sl = wrapper_sl.raw_dataset
    n_sl = len(wrapper_sl)
    occ_s0 = torch.stack([raw_sl[i][0][0][0] for i in range(n_sl)])
    occ_s1 = torch.stack([raw_sl[i][1] for i in range(n_sl)])
    actions_sl = torch.stack([raw_sl.get_raw_action(i) for i in range(n_sl)])
    ep_sl = torch.tensor([raw_sl.get_run_index(i) for i in range(n_sl)])
    s_sl, e_sl = actions_to_pixels(actions_sl, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    print(f"\nslates: {n_sl} transitions, {ep_sl.unique().numel()} slates "
          f"(from {args.slate_cfg})")

    X_sl = torch.stack([wrapper_sl[i]["input"] for i in range(n_sl)])
    P_sl = torch.stack([wrapper_sl[i]["physics"] for i in range(n_sl)])
    unet_pred_sl = unet_forward(model, X_sl, P_sl)

    control_preds = {
        "persistence": occ_s0,
        "mean-delta": predict_meandelta(bmd, occ_s0, s_sl, e_sl, R, (H, W), CR),
        "linear (ridge->identity)": predict_world(A, occ_s0, s_sl, e_sl, R, (H, W), CR),
        "UNet": unet_pred_sl,
    }

    results = {"accuracy": accuracy, "control": {}}
    for goal in ("corner", "center"):
        dw = lyapunov_weights((H, W), goal, "cpu")
        v0 = lyapunov(occ_s0, dw)
        v1 = lyapunov(occ_s1, dw)
        dv_true = v1 - v0
        print(f"\n=== goal={goal} === dv_true mean={float(dv_true.mean()):+.5f} "
              f"sd={float(dv_true.std()):.5f} helpful(dV<0)="
              f"{100 * float((dv_true < 0).float().mean()):.0f}%")
        print(f"{'model':28s}{'n_slt':>6s}{'spearman':>11s}{'(sd)':>8s}"
              f"{'sign%':>8s}{'slate4':>9s}{'(sd)':>8s}")
        goal_out = {}
        for name, pred in control_preds.items():
            dv_pred = lyapunov(pred, dw) - v0
            m = per_slate_metrics(dv_pred, dv_true, ep_sl, min_slate=args.min_slate)
            goal_out[name] = m
            if m["n_slates"] == 0:
                print(f"  {name:26s}  (no slate had >= {args.min_slate} live candidates)")
                continue
            print(f"  {name:26s}{m['n_slates']:6d}{m['spearman_mean']:11.3f}"
                  f"{m['spearman_sd']:8.3f}{100 * m['sign_mean']:7.0f}%"
                  f"{m['slate4_mean']:9.3f}{m['slate4_sd']:8.3f}")
        # oracle: dv_pred := dv_true exactly (the ranking ceiling)
        m = per_slate_metrics(dv_true.clone(), dv_true, ep_sl, min_slate=args.min_slate)
        goal_out["oracle"] = m
        print(f"  {'oracle':26s}{m['n_slates']:6d}{m['spearman_mean']:11.3f}"
              f"{m['spearman_sd']:8.3f}{100 * m['sign_mean']:7.0f}%"
              f"{m['slate4_mean']:9.3f}{m['slate4_sd']:8.3f}")
        # PAIRED comparison. The models are scored on the SAME slates, whose
        # difficulty varies a lot, so the across-slate sd printed above is the
        # wrong floor for a between-model difference -- it is dominated by
        # slate difficulty that BOTH models share. EXP-0016 made exactly this
        # correction for C-008 and the verdict changed.
        import numpy as _np
        base = goal_out.get("linear (ridge->identity)", {}).get("per_slate", {})
        for name, m in goal_out.items():
            ps = m.get("per_slate") or {}
            common = sorted(set(ps) & set(base))
            if name.startswith("linear") or len(common) < 5:
                continue
            d = _np.array([ps[k]["slate4"] - base[k]["slate4"] for k in common])
            sem = d.std(ddof=1) / _np.sqrt(len(d))
            print(f"  PAIRED vs linear: {name:22s} mean_diff={d.mean():+.4f} "
                  f"paired_sd={d.std(ddof=1):.4f} sem={sem:.4f} "
                  f"t={d.mean()/max(sem,1e-9):+.2f} n={len(d)} "
                  f"wins={int((d>0).sum())}/{len(d)}")
        results["control"][goal] = {k: {kk: vv for kk, vv in v.items()
                                        if kk != "per_slate"}
                                    for k, v in goal_out.items()}

    print("\npersistence's dv_pred is 0 for every candidate by construction: "
          "it cannot rank at all, and its 'spearman'/'slate4' numbers above "
          "are noise around 0, not a meaningful correlation -- report as "
          "'cannot rank', not as a score.")

    out = {"tag": args.tag, "dataset_cfg": args.dataset_cfg,
           "slate_cfg": args.slate_cfg, "run_dir": args.run_dir,
           "n_train": int(data_tr.occ_t.shape[0]), "n_test_image": n_te,
           "n_slate_transitions": n_sl, "n_slates": int(ep_sl.unique().numel()),
           "res": R, "crop": CR, "ridge": RIDGE, "results": results}
    out_path = args.out or f"runs_exp0024/control_eval_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
