"""EXP-0061 final test evaluation -- per-row / per-slate outputs for every model.

Composes Baselines/common/eval_report.py's own functions (no fork of the metric
code), on corpus `flex_ds0019_mask` with truth = cell.occ1 = the BINARY image
mask of the true after-state (= `eval_report.py --truth-scoring image`, i.e.
`_capture_report(..., truth_s0=None)`). What it adds over eval_report.py's JSON:

  * per-row swept-region rms of prediction and of persistence (so accuracy CIs
    and strata are ratios of population means, never means of ratios);
  * per-slate capture for every model incl. a per-slate random floor;
  * GNN node-sampling noise: gnn_flex_drp re-run with 3 FPS start seeds
    (seed offset 0 = the registered predictor, bit-identical to eval_report);
  * DS-0020 val per-row errors for the same models (val-vs-test gap).

Outputs (atomic, rewritten after every model):
  results/final_eval/ds0019_rows.npz   per-row errors + row metadata
  results/final_eval/ds0019.json       accuracy + capture (per_slate) per model
  results/final_eval/ds0020val_rows.npz, ds0020val.json

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py ds0019
    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py ds0020_val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from Baselines.common import eval_report as er  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from fit_linear_foresight import actions_to_pixels, metrics, plate_width_px, swept_region_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from score_nfd_flex import degeneracy, nfd_spec  # noqa: E402

OUT = Path("experiments/EXP-0061-flex-cross-corpus-rerun/results/final_eval")
D19 = Path("datasets/DS-0019-slates-flex-pile-varN")
D20 = "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/config.yaml"
NFD_CKPT = "Baselines/NFD/runs/nfd_3ch_flex_mask_seed{}/unet_best.pth"


def region_of(cell):
    H, W = cell.occ0.shape[-2:]
    s, e = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    p = plate_width_px(cell.raw, W)
    return swept_region_mask(s, e, (H, W), 0.5 * p + 2.0, 0.5 * p)


def row_rms(pred, truth, region):
    """per-row swept-region rms, exactly metrics()'s per-row term."""
    n = pred.shape[0]
    w = region.reshape(n, -1)
    npix = w.sum(1).clamp_min(1.0)
    d = ((pred - truth) * region).reshape(n, -1)
    return (d.pow(2).sum(1) / npix).sqrt()


def gnn_with_sampling_seed(offset: int):
    """The registered gnn_flex_drp predictor with its per-state FPS seed shifted
    by `offset` (offset 0 = unchanged). Monkeypatches only the module-level
    `perceive` the predictor calls; restored by the next call."""
    import Baselines.GNN.flex_predictor as fp
    if not hasattr(fp, "_perceive_orig"):
        fp._perceive_orig = fp.perceive
    orig = fp._perceive_orig
    fp.perceive = (orig if offset == 0 else
                   (lambda c, d, py, n, seed: orig(c, d, py, n, (seed + 7919 * offset) % (2 ** 32))))
    spec = er.MODELS["gnn_flex_drp"]
    return spec, er._load_predictor(spec)


def model_list(which):
    ms = [(f"nfd_s{s}", "nfd", s) for s in range(3)]
    ms += [("lf_flex_switched", "lf", None), ("lf_flex_single", "lf", None)]
    ms += [(f"gnn_flex_drp_samp{k}", "gnn", k) for k in (range(3) if which == "ds0019" else range(1))]
    ms += [("gnn_flex_drp_n50", "gnn_n", 50)]   # sensitivity: N small enough that no state needs the fallback
    return ms


def build(kind, arg, name):
    if kind == "nfd":
        spec = nfd_spec(NFD_CKPT.format(arg))
        return spec, er._load_predictor(spec)
    if kind == "lf":
        spec = er.MODELS[name]
        return spec, er._load_predictor(spec)
    if kind == "gnn_n":
        gnn_with_sampling_seed(0)
        spec = dict(er.MODELS["gnn_flex_drp"], kwargs=dict(er.MODELS["gnn_flex_drp"]["kwargs"], particle_num=arg))
        return spec, er._load_predictor(spec)
    return gnn_with_sampling_seed(arg)


def save(res, rows, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    er._write_json_atomic(str(OUT / f"{tag}.json"), res)
    tmp = OUT / f"{tag}_rows.tmp.npz"
    np.savez(tmp, **rows)
    os.replace(tmp, OUT / f"{tag}_rows.npz")


def slate_meta():
    init_pos = {}
    for line in open(D19 / "manifest.jsonl"):
        r = json.loads(line)
        if r.get("type") == "state_init":
            init_pos[int(r["state_idx"])] = r.get("init_pos")
    nrig, npart = {}, {}
    for s in init_pos:
        z = np.load(D19 / "cache" / f"state_{s:03d}.npz")
        nrig[s], npart[s] = int(z["n_rigids"]), int(z["n_particles"])
    return init_pos, nrig, npart


def run(which, only=None):
    t0 = time.time()
    if which == "ds0019":
        cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    else:
        cell = load_flex_cell(D20, "val", tag="ds0020_val_mask", occ_source="image_mask")
    print(f"[{which}] {len(cell.occ0)} rows ({time.time() - t0:.0f}s)", flush=True)
    region = region_of(cell)
    truth, prev = cell.occ1.float(), cell.occ0.float()
    rows = dict(slate=cell.slate_idx.numpy(), step=cell.step_idx.numpy(),
                push_length=cell.push_length.numpy(),
                rms_persistence=row_rms(prev, truth, region).numpy())
    res = dict(corpus=which, truth="binary image mask of the true after-state (cell.occ1, occ_source=image_mask)",
               n_rows=int(len(cell.occ0)), models={})
    if only and (OUT / f"{which}.json").exists():   # merge new models into the existing outputs
        res = json.load(open(OUT / f"{which}.json"))
        rows = dict(np.load(OUT / f"{which}_rows.npz"))
    elif which == "ds0019":
        ip, nr, npt = slate_meta()
        sl = rows["slate"]
        rows["init_pos_blob"] = np.array([ip[int(s)] == "rand_blob" for s in sl])
        rows["n_rigids"] = np.array([nr[int(s)] for s in sl])
        rows["n_particles"] = np.array([npt[int(s)] for s in sl])
        res["slate_meta"] = {int(s): dict(init_pos=ip[s], n_rigids=nr[s], n_particles=npt[s]) for s in ip}
        res["degeneracy"] = degeneracy(cell)
        res["models"]["persistence"] = dict(
            accuracy=metrics(prev, truth, prev, region=region)["accuracy"],
            capture=er._capture_report(cell, cell.occ0, "default", None))
        # per-slate random floor: mean over 20 noise seeds (eval_report's own seeds)
        reps = []
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            reps.append(er._capture_report(cell, torch.rand(cell.occ0.shape, generator=g), "default", None))
        cap = reps[0]
        cap["per_slate"] = {gl: {v: np.mean([r["per_slate"][gl][v] for r in reps], 0).tolist()
                                 for v in er.VALUE_FNS} for gl in cap["per_slate"]}
        cap["per_goal"] = {gl: {v: float(np.mean(cap["per_slate"][gl][v])) for v in er.VALUE_FNS}
                           for gl in cap["per_goal"]}
        cap["averaged_over_goals"] = {v: float(np.mean([cap["per_goal"][gl][v] for gl in cap["per_goal"]]))
                                      for v in er.VALUE_FNS}
        res["models"]["random"] = dict(accuracy=None, capture=cap, note="per-slate mean over 20 noise seeds")
        save(res, rows, which)
    for name, kind, arg in model_list(which):
        if only and name not in only:
            continue
        t1 = time.time()
        spec, pred_ = build(kind, arg, name)
        acc, pred = er._accuracy(spec, pred_, cell, "cuda")
        rows[f"rms_{name}"] = row_rms(pred.float(), truth, region).numpy()
        m = dict(accuracy=acc, seconds=time.time() - t1,
                 accuracy_check=float(1 - rows[f"rms_{name}"].mean() / rows["rms_persistence"].mean()))
        if which == "ds0019":
            m["capture"] = er._capture_report(cell, pred, "default", None)
            print(name, f"acc {acc:.4f}", {k: round(v, 4) for k, v in m["capture"]["averaged_over_goals"].items()},
                  f"{m['seconds']:.0f}s", flush=True)
        else:
            print(name, f"val acc {acc:.4f} {m['seconds']:.0f}s", flush=True)
        res["models"][name] = m
        save(res, rows, which)
        del pred
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ds0019", "ds0020_val"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.which, a.only)
