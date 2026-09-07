"""EXP-0026 stage 1: cache dV predictions for the selection-pressure curve.

Every control number in the register is `slate4` -- pick the best of 4
candidates drawn at random from a 32-action same-state slate. Real MPC ranks
hundreds to thousands. This stage recomputes EXP-0024's per-candidate
dV_pred / dV_true vectors on `Genesis/data/slates/n20_heap_5mm` (identical
pipeline, identical fit, identical checkpoint) and caches them, so stage 2
(`exp0026_kcurve.py`) can sweep the slate size K without touching the GPU or
refitting anything.

Nothing about the models changes here. The only reason this is a separate
script from `scripts/probes/exp0024_control_eval.py` is that EXP-0024 threw the
per-candidate vectors away and reported only per-slate summaries.

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0026_selection_pressure.py \
        configs/dataset/genesis_cube_spectrum_n20.yaml \
        configs/dataset/genesis_cube_spectrum_n20_slates.yaml \
        runs_exp0024/unetfilm_cube_spectrum_n20 \
        --out runs_exp0026/dv_cache.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, predict_world,
)
from registry.dataset_registry import build_dataset
from scripts.probes.exp0009_rerun import predict_meandelta
from scripts.probes.exp0021_eval import load_unet, unet_forward
from control_utility_test import lyapunov, lyapunov_weights
from utils import git_provenance

R, CR, RIDGE = 64, 1.0, 1.0  # EXP-0024's fit for this dataset/slate pairing


def _slate_files(scfg, split):
    """The data files this config/split actually loads, in run-index order.

    Recorded in the cache so a slate id can be traced back to a file: the
    registry's split assignment withholds one group whatever test_pct is set
    to (see configs/dataset/genesis_cube_spectrum_n20_slates_all.yaml), and
    which one it is depends on an md5 ordering rather than on the file name.
    """
    from Genesis.training.dataset import PileSweepData
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent / "Genesis" / "data"
    files = []
    for path in scfg["paths"]:
        full = root / path
        runs = PileSweepData._collect_run_paths(PileSweepData, full)
        runs = PileSweepData._filter_split(runs, split, scfg.get("val_pct", 5),
                                           scfg.get("test_pct", 5))
        files.extend(str(d.name) for d, _ in runs)
    return files


def _degradation_arms(linear_pred, occ0, occ1, s_px, e_px, grid):
    """EXP-0025's degradation spectrum, recipes unchanged.

    `scripts/probes/full_spectrum_same_state.py` builds these from the linear
    operator's predicted delta and reports slate4 for each; the only reason
    they are rebuilt here is that its numbers came from the
    occupancy_foresight loader, while everything in EXP-0026 runs through the
    registry/PileSweepData path. Same recipes, same levels, same noise seed
    (1234), so the K-dependence is measured within one code path even though
    the absolute values are not identical to EXP-0025's.
    """
    from fit_linear_foresight import swept_region_mask
    from transforms.particle_fields import _gaussian_blur2d
    from scripts.probes.fss_scale_decomp import build_pyramid
    from scripts.probes.same_state_degradation import finest_band_noise

    H, W = grid
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0,
                               0.5 * plate_px)
    sig_laps, _ = build_pyramid(occ1 - occ0, 4)
    region_std = float(sig_laps[0][region > 0].std())
    print(f"finest-band true-signal std over region: {region_std:.5f}")

    delta = linear_pred - occ0
    gen = torch.Generator().manual_seed(1234)
    arms = {}

    def field(d):
        return (occ0 + d).clamp(0.0, 1.0)

    for k in (1, 2, 4):
        arms[f"displacement k={k}"] = field(torch.roll(delta, shifts=k, dims=-1))
    for a in (0.5, 0.75, 1.25, 1.5):
        arms[f"amplitude a={a}"] = field(a * delta)
    for m in (0.5, 1.0, 2.0):
        arms[f"hf-noise m={m}"] = field(delta + finest_band_noise(delta.shape, gen,
                                                                 region_std * m))
    for sig in (1.0, 2.0):
        arms[f"blur s={sig}"] = field(_gaussian_blur2d(delta, sig))
    arms["wrong-physics"] = field(torch.roll(delta, shifts=1, dims=0))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_cfg")
    ap.add_argument("slate_cfg")
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="train",
                    help="split to load the slate config under. The default "
                         "'train' with val_pct=0/test_pct=0 is the ONLY "
                         "setting that yields all 50 slates -- see "
                         "configs/dataset/genesis_cube_spectrum_n20_slates_all.yaml. "
                         "Pass 'test' with the sibling _slates.yaml to "
                         "reproduce EXP-0024's 49-slate subset.")
    ap.add_argument("--degradations", action="store_true",
                    help="also cache EXP-0025's degradation arms (displacement, "
                         "amplitude, hf-noise, blur, wrong-physics) applied to "
                         "the linear operator's predicted delta, so the K-sweep "
                         "can re-test 'blur is nearly free' (C-035) at slate "
                         "sizes above 4. Recipes are imported unchanged from "
                         "the EXP-0025 probe; only the code path that produces "
                         "the fields differs (registry/PileSweepData here).")
    ap.add_argument("--out", default="runs_exp0026/dv_cache.pt")
    ap.add_argument("--goals", default="corner,center",
                     help="comma-separated lyapunov_weights goal keys to "
                          "cache dV for. Default is EXP-0024/EXP-0026's "
                          "original two goals, unchanged. Pass additional "
                          "keys (e.g. ind-square8, distclip-corner-r4) to "
                          "extend the cache without touching the original "
                          "corner/center numbers -- lyapunov_weights keeps "
                          "those two byte-identical to before.")
    args = ap.parse_args()

    prov = git_provenance()
    print(f"provenance: {prov}")

    # ---- fit the linear baselines on the training split (EXP-0024's fit) ----
    data_tr = load_transition_arrays(args.dataset_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ_t.shape[0]
    print(f"train: {n_tr} transitions, grid {H}x{W}")

    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)

    model, _, missing = load_unet(args.run_dir)
    print(f"UNet load: {missing}")

    # ---- slate predictions -------------------------------------------------
    scfg = yaml.safe_load(open(args.slate_cfg).read())
    wrapper = build_dataset(scfg, args.split)
    raw = wrapper.raw_dataset
    n = len(wrapper)
    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    ep = torch.tensor([raw.get_run_index(i) for i in range(n)])
    s_px, e_px = actions_to_pixels(actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    slate_files = _slate_files(scfg, args.split)
    print(f"slates: {n} transitions, {ep.unique().numel()} slate groups "
          f"({len(slate_files)} files in split '{args.split}')")

    X = torch.stack([wrapper[i]["input"] for i in range(n)])
    P = torch.stack([wrapper[i]["physics"] for i in range(n)])

    preds = {
        "persistence": occ0,
        "mean-delta": predict_meandelta(bmd, occ0, s_px, e_px, R, (H, W), CR),
        "linear": predict_world(A, occ0, s_px, e_px, R, (H, W), CR),
        "UNet": unet_forward(model, X, P),
    }

    if args.degradations:
        preds.update(_degradation_arms(preds["linear"], occ0, occ1,
                                       s_px, e_px, (H, W)))
        print(f"degradation arms: {len(preds) - 4} added")

    out = {"ep": ep, "actions": actions, "provenance": prov,
           "slate_files": slate_files, "split": args.split,
           "config": {"dataset_cfg": args.dataset_cfg, "slate_cfg": args.slate_cfg,
                      "run_dir": args.run_dir, "R": R, "crop": CR, "ridge": RIDGE},
           "dv": {}}
    for goal in [g.strip() for g in args.goals.split(",") if g.strip()]:
        dw = lyapunov_weights((H, W), goal, "cpu")
        v0 = lyapunov(occ0, dw)
        dv_true = lyapunov(occ1, dw) - v0
        g = {"dv_true": dv_true, "v0": v0}
        for name, pred in preds.items():
            g[name] = lyapunov(pred, dw) - v0
        out["dv"][goal] = g
        print(f"goal={goal}: dv_true mean {float(dv_true.mean()):+.5f} "
              f"sd {float(dv_true.std()):.5f} "
              f"helpful {100 * float((dv_true < 0).float().mean()):.0f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
