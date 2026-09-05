"""EXP-0021: swept-region head-to-head, one cell.

For one trained UNetFilm checkpoint (`run_dir`, produced by
`scripts/probes/exp0021_train_all.sh` from a `configs/training/exp0021_*.yaml`
config), computes swept-region rms (as % of persistence) for:

  persistence, mean-delta, identity(warp only), linear-ridge1.0 (ridge toward
  identity, res=64/crop=0.5 -- the EXP-0018 headline cell), UNet true/shuffled/
  zeroed action -- each reported twice: all held-out transitions, and
  contact>0 only.

Reuses fit_linear_foresight.py's own canonicalise/fit_operator/predict_world/
metrics/swept_region_mask/contact_score and exp0009_rerun.py's mean-delta
helper verbatim -- no new rasteriser, no new warp, no new metric. The linear
operator is fit on the TRAIN split of the SAME dataset config (registry's own
file-level split, deterministic md5-hash-of-path bucketing -- identical rule
to the one that produced the UNet's own train/val/test partition), and scored
on the TEST split -- i.e. the linear operator and the UNet are evaluated on
literally the same held-out transitions, not two different splits.

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0021_eval.py \
        configs/training/exp0021_unetfilm_blind_n5.yaml \
        runs_granularity/unetfilm_blind_n5 --tag blind_n5
"""
from __future__ import annotations

import argparse
import json

import torch
import yaml

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, contact_score, fit_operator, metrics,
    predict_world, swept_region_mask,
)
from registry.dataset_registry import build_dataset
from registry.model_registry import build_model
from scripts.probes.exp0009_rerun import predict_meandelta

R, CR = 64, 0.5           # EXP-0018's headline cell: res=64, crop=0.5
RIDGE = 1.0                # ridge toward identity (EXP-0016/0018 "single ridge1")


def load_unet(run_dir: str):
    cfg = yaml.safe_load(open(f"{run_dir}/run_config.yaml").read())
    model = build_model(cfg["model"])
    sd = torch.load(f"{run_dir}/unet_best.pth", map_location="cpu",
                    weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg, missing


def unet_forward(model, X, P):
    with torch.no_grad():
        out = model({"input": X, "physics": P})
        if not torch.is_tensor(out):
            out = out["prediction"]
        return torch.sigmoid(out).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_cfg")
    ap.add_argument("run_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the shuffle-ablation permutation")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    print(f"=== EXP-0021 eval: {args.tag} ===")

    # ---- train split: fit the linear baselines ---------------------------
    data_tr = load_transition_arrays(args.dataset_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    print(f"train: {data_tr.occ_t.shape[0]} transitions, grid {H}x{W}")

    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR
                      ).reshape(data_tr.occ_t.shape[0], -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR
                      ).reshape(data_tr.occ_t.shape[0], -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE)
    bmd = (Y1 - Y0).mean(dim=1)

    # ---- test split: everything is scored here ----------------------------
    cfg = yaml.safe_load(open(args.dataset_cfg).read())
    wrapper = build_dataset(cfg, "test")
    raw = wrapper.raw_dataset
    n = len(wrapper)
    print(f"test: {n} transitions")

    occ_t = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ_t1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    s_te, e_te = actions_to_pixels(actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))

    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_te, e_te, (H, W), half_width_px=0.5 * plate_px + 2.0,
                               pad_px=0.5 * plate_px)
    contact = contact_score(occ_t, s_te, e_te, (H, W), plate_px)
    strata = {"all": torch.ones(n, dtype=torch.bool), "contact>0": contact > 0}
    for name, m in strata.items():
        print(f"  stratum {name}: n={int(m.sum())}")

    preds = {
        "persistence": occ_t,
        "mean-delta": predict_meandelta(bmd, occ_t, s_te, e_te, R, (H, W), CR),
        "identity(warp only)": predict_world(
            torch.eye(R * R), occ_t, s_te, e_te, R, (H, W), CR),
        "linear-ridge1.0": predict_world(A, occ_t, s_te, e_te, R, (H, W), CR),
    }

    # ---- UNet: true / shuffled / zeroed action ----------------------------
    model, run_cfg, missing = load_unet(args.run_dir)
    print(f"  UNet load: {missing}")
    X = torch.stack([wrapper[i]["input"] for i in range(n)])
    P = torch.stack([wrapper[i]["physics"] for i in range(n)])
    preds["unet-true"] = unet_forward(model, X, P)

    torch.manual_seed(args.seed)
    perm = torch.randperm(n)
    Xs = X.clone()
    Xs[:, 1] = X[perm, 1]
    preds["unet-shuffled"] = unet_forward(model, Xs, P)

    Xz = X.clone()
    Xz[:, 1] = 0.0
    preds["unet-zeroed"] = unet_forward(model, Xz, P)

    # ---- score every prediction on every stratum --------------------------
    results = {}
    for stratum_name, smask in strata.items():
        idx = smask.nonzero(as_tuple=True)[0]
        region_s = region[idx]
        occ_s, occ1_s = occ_t[idx], occ_t1[idx]
        rp = None
        results[stratum_name] = {}
        for name, pred in preds.items():
            m = metrics(pred[idx], occ1_s, occ_s, region=region_s)
            results[stratum_name][name] = m
            if name == "persistence":
                rp = m["rms"]
        print(f"\n=== {args.tag} / {stratum_name} (n={len(idx)}), swept-region ===")
        for name, m in results[stratum_name].items():
            print(f"  {name:24s} rms={m['rms']:.5f}  "
                  f"pct_persist={100 * m['rms'] / rp:6.1f}%  "
                  f"explained={m['explained']:+.4f}")

    out = {"tag": args.tag, "dataset_cfg": args.dataset_cfg,
          "run_dir": args.run_dir, "n_train": int(data_tr.occ_t.shape[0]),
          "n_test": n, "res": R, "crop": CR, "ridge": RIDGE,
          "results": results}
    out_path = args.out or f"runs_granularity/exp0021_eval_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
