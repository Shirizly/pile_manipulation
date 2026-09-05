"""Is the UNet's win over the linear operator an artifact of denying it blur?

EXP-0021 scored both at sigma=0. But C-002 says the SE(2) warp the linear
method depends on destroys more than a push changes unless the field is
smoothed to sigma~1 -- and measured on L040 the operator goes 77.6% -> 57.8%
of persistence with blur. So the operator was run in the one configuration a
standing claim says is broken for it.

Those two numbers are NOT comparable: they score different targets against
different persistence baselines. The fair test is to score both models on the
SAME target, letting each preprocess internally as it likes:

    linear : fit and predict on the blurred field (as C-002 says it needs)
    UNet   : takes the sharp input it was trained on, predicts sharp, and its
             prediction is then blurred for scoring
    truth  : blurred identically for both

That measures "does the model get the coarse structure right", which is the
same question for both, and it is the comparison neither EXP-0018 nor EXP-0021
ran.

    PYTHONPATH=. python scripts/probes/exp0022_blur_fair.py \
        configs/dataset/genesis_granularity_blind_n50.yaml \
        runs_granularity/unetfilm_blind_n50 --tag blind_n50
"""
from __future__ import annotations

import argparse

import torch

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask,
)
from scripts.probes.exp0021_eval import load_unet, unet_forward
import yaml

R, CR, RIDGE = 64, 0.5, 1.0


def gblur(x, sig):
    if sig <= 0:
        return x
    k = int(2 * round(3 * sig) + 1)
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sig * sig)); g = g / g.sum()
    y = torch.nn.functional.conv2d(x.unsqueeze(1), g.view(1, 1, 1, -1), padding=(0, k // 2))
    return torch.nn.functional.conv2d(y, g.view(1, 1, -1, 1), padding=(k // 2, 0)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_cfg"); ap.add_argument("run_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.dataset_cfg).read())
    d_tr = load_transition_arrays(a.dataset_cfg, split="train")
    d_te = load_transition_arrays(a.dataset_cfg, split="test")
    H, W = d_tr.occ_t.shape[-2:]

    model, mcfg, _ = load_unet(a.run_dir)
    from registry.dataset_registry import build_dataset
    raw = build_dataset(cfg, "test")
    X = torch.stack([raw[i]["input"] for i in range(len(raw))])
    P = torch.stack([raw[i]["physics"] for i in range(len(raw))])
    truth = torch.stack([raw[i]["target"] for i in range(len(raw))])
    occ0 = X[:, 0]
    pred_unet = unet_forward(model, X, P)

    s_tr, e_tr = actions_to_pixels(d_tr.actions, d_tr.workspace_min, d_tr.workspace_max, (H, W))
    s_te, e_te = actions_to_pixels(d_te.actions, d_te.workspace_min, d_te.workspace_max, (H, W))
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(s_te, e_te, (H, W), 0.5 * plate + 2.0, 0.5 * plate)

    print(f"=== {a.tag}: same target for both models, sigma={a.sigma} ===")
    print(f"{'model':30s} {'rms':>9s} {'% of change':>12s}")

    for label, sig_fit in [("scored SHARP (EXP-0021 setting)", 0.0),
                           (f"scored BLURRED sigma={a.sigma}", a.sigma)]:
        t_b = gblur(truth, sig_fit)
        o0_b = gblur(occ0, sig_fit)
        base = metrics(o0_b, t_b, o0_b, region=region)["rms"]
        # linear: fit and predict in whatever field it is being scored on
        tr0, tr1 = gblur(d_tr.occ_t, sig_fit), gblur(d_tr.occ_t1, sig_fit)
        n_tr = tr0.shape[0]
        Y0 = canonicalise(tr0, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
        Y1 = canonicalise(tr1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
        A = fit_operator(Y0, Y1, RIDGE, toward_identity=True)
        p_lin = predict_world(A, o0_b, s_te, e_te, R, (H, W), CR)
        # UNet: sharp input as trained; its prediction is blurred for scoring
        p_unet = gblur(pred_unet, sig_fit)
        print(f"  -- {label}")
        for nm, p in [("persistence", o0_b), ("linear-ridge1.0", p_lin),
                      ("UNet (blurred for scoring)", p_unet)]:
            r = metrics(p, t_b, o0_b, region=region)["rms"]
            print(f"  {nm:28s} {r:9.5f} {100 * r / base:11.1f}%")


if __name__ == "__main__":
    main()
