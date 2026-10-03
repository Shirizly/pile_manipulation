"""EXP-0061 -- score an NFD checkpoint trained on DS-0020 image masks.

Composes Baselines/common/eval_report.py's own functions (no fork of the metric
code). Ground truth = the BINARY IMAGE MASK of the true after-state (EXP-0061
user decision): the cell is loaded with occ_source="image_mask", so cell.occ1 IS
the mask, and _capture_report is called with truth_s0=None (its "image" truth
path = cell.occ1). Model input occ0 is the mask too.

    # DS-0019 (100 same-state slates): slateN (random_quadrant/ring_O/T x
    # lyapunov/mass_in_region/signed_mass) + swept-region accuracy, with
    # persistence and random rows, per-slate capture kept, dv_true==0 fractions
    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/score_nfd_flex.py ds0019 \
        --ckpt Baselines/NFD/runs/nfd_3ch_flex_mask_seed0/unet_best.pth \
        --out experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_seed0_ds0019.json

    # DS-0020 val: swept-region accuracy vs persistence only (several ckpts)
    python -u .../score_nfd_flex.py ds0020_val --ckpt A --ckpt B ... --out .../nfd_val_accuracy.json

Results are rewritten atomically after every unit (model / reference row).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from Baselines.common import eval_report as er  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from fit_linear_foresight import actions_to_pixels, metrics, plate_width_px, swept_region_mask  # noqa: E402

D20 = "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/config.yaml"


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    er._write_json_atomic(str(path), obj)


def region_of(cell):
    H, W = cell.occ0.shape[-2:]
    s, e = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    p = plate_width_px(cell.raw, W)
    return swept_region_mask(s, e, (H, W), 0.5 * p + 2.0, 0.5 * p), p


def nfd_spec(ckpt):
    return dict(er.MODELS["nfd_randlen"], ckpt=ckpt)


def parity_check(predictor, cell, ckpt, n=16):
    """predictor.predict_occ (eval harness path) vs the training-path tensors
    (FlexPileData.__getitem__ -> UNet -> sigmoid) on n rows: max |diff|."""
    from registry.model_registry import build_model
    import yaml
    import Baselines.NFD.nfd_lib  # noqa: F401
    cfg = yaml.safe_load(open(Path(ckpt).parent / "run_config.yaml"))
    mw = build_model(cfg["model"]).cuda().eval()
    mw.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True))
    idx = list(range(0, len(cell.raw), max(1, len(cell.raw) // n)))[:n]
    xs = torch.stack([cell.raw[i][0][0] for i in idx]).cuda()
    with torch.no_grad():
        direct = torch.sigmoid(mw.model(xs)).squeeze(1).cpu()
    via = er._predict(predictor, _subcell(cell, idx), "cuda")
    return float((direct - via).abs().max())


def _subcell(cell, idx):
    import dataclasses
    t = torch.as_tensor(idx)
    kw = {}
    for f in dataclasses.fields(cell):
        v = getattr(cell, f.name)
        kw[f.name] = v[t] if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == cell.occ0.shape[0] else v
    return type(cell)(**kw)


def degeneracy(cell, goal_set="default"):
    """Per goal x value fn: fraction of rows with dv_true == 0 (true after-state
    value == before-state value, on the binary-mask truth) and fraction of slates
    whose whole pool has ONE true value (slateN undefined / NaN there)."""
    H, W = cell.occ0.shape[-2:]
    fns = {"lyapunov": lambda x, m, d: er.lyapunov(x, d),
           "mass_in_region": lambda x, m, d: er.mass_in_region(x, m),
           "signed_mass": lambda x, m, d: er.signed_mass_in_region(x, m)}
    out = {g: {v: {"dv0_rows": 0, "rows": 0, "flat_slates": 0, "slates": 0} for v in fns}
           for g in er.goal_names(goal_set)}
    cache = {}
    for sid in cell.slate_idx.unique().tolist():
        rows = (cell.slate_idx == sid).nonzero(as_tuple=True)[0]
        o0, o1 = cell.occ0[rows].float(), cell.occ1[rows].float()
        for g, m, d in er._goals_for_slate(goal_set, H, W, sid, cache):
            for v, f in fns.items():
                vt, v0 = f(o1, m, d), f(o0, m, d)
                dv = (vt - v0).abs()
                c = out[g][v]
                c["dv0_rows"] += int((dv < 1e-9).sum()); c["rows"] += len(rows)
                c["flat_slates"] += int(float(vt.max() - vt.min()) < 1e-9); c["slates"] += 1
    return {g: {v: dict(frac_dv_true_zero=c["dv0_rows"] / c["rows"],
                        frac_slates_flat=c["flat_slates"] / c["slates"]) for v, c in d.items()}
            for g, d in out.items()}


def run_ds0019(args):
    t0 = time.time()
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    print(f"loaded {len(cell.occ0)} rows, {len(cell.slate_idx.unique())} slates ({time.time() - t0:.0f}s)", flush=True)
    region, plate_px = region_of(cell)
    res = dict(corpus="DS-0019 (flex_ds0019_mask: occ0 AND truth = binary image mask)",
               truth="binary image mask of the true after-state (cell.occ1, occ_source=image_mask); "
                     "eval_report._capture_report(truth_s0=None)",
               n_rows=int(len(cell.occ0)), n_slates=int(len(cell.slate_idx.unique())),
               plate_width_px=plate_px, goal_set="default", rows={})
    res["degeneracy"] = degeneracy(cell)
    # SECONDARY (not the user-decided truth): slateN against the soft particle
    # splat (eval_report's default --truth-scoring soft), for comparison only.
    soft = er.truth_for_scoring(cell, (cell.step_idx == 0).nonzero(as_tuple=True)[0])
    res["secondary_soft_particle_truth"] = {
        "persistence": er._capture_report(cell, cell.occ0, "default", soft)["averaged_over_goals"]}
    acc_p = metrics(cell.occ0.float(), cell.occ1.float(), cell.occ0.float(), region=region)["accuracy"]
    res["rows"]["persistence"] = dict(accuracy=acc_p, capture=er._capture_report(cell, cell.occ0, "default", None),
                                      note="slateN row DEGENERATE (constant prediction)")
    res["rows"]["random"] = dict(accuracy=None, capture=er._random_floor_capture(cell, goal_set="default", truth_s0=None))
    _write(args.out, res)
    for ck in args.ckpt:
        name = args.name or Path(ck).parent.name
        pred_ = er._load_predictor(nfd_spec(ck))
        t1 = time.time()
        acc, pred = er._accuracy(nfd_spec(ck), pred_, cell, "cuda")
        cap = er._capture_report(cell, pred, "default", None)
        res["rows"][name] = dict(ckpt=ck, accuracy=acc, capture=cap, parity_max_abs=parity_check(pred_, cell, ck),
                                 seconds=time.time() - t1)
        res["secondary_soft_particle_truth"][name] = er._capture_report(cell, pred, "default", soft)["averaged_over_goals"]
        print(name, "acc", acc, cap["averaged_over_goals"], flush=True)
        _write(args.out, res)
    return res


def run_val(args):
    cell = load_flex_cell(D20, "val", tag="ds0020_val_mask", occ_source="image_mask")
    region, plate_px = region_of(cell)
    res = dict(corpus="DS-0020 val (splits.json, 200 trajectories), occ0 AND truth = binary image mask",
               n_rows=int(len(cell.occ0)), plate_width_px=plate_px, rows={})
    res["rows"]["persistence"] = dict(accuracy=metrics(cell.occ0.float(), cell.occ1.float(), cell.occ0.float(),
                                                       region=region)["accuracy"])
    for ck in args.ckpt:
        name = Path(ck).parent.name + "/" + Path(ck).stem
        p = er._load_predictor(nfd_spec(ck))
        acc, pred = er._accuracy(nfd_spec(ck), p, cell, "cuda")
        m = metrics(pred, cell.occ1.float(), cell.occ0.float(), region=region)
        m_full = metrics(pred, cell.occ1.float(), cell.occ0.float())
        res["rows"][name] = dict(ckpt=ck, accuracy=acc, rms_swept=m["rms"], soft_iou_swept=m["soft_iou"],
                                 accuracy_whole_grid=m_full["accuracy"])
        print(name, res["rows"][name], flush=True)
        _write(args.out, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ds0019", "ds0020_val"])
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run_ds0019(args) if args.which == "ds0019" else run_val(args)
    _write(args.out, res)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
