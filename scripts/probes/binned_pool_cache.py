"""Build a dV cache for a binned slate corpus, so `pool_inspect.py` /
`pool_survey.py` can be pointed at `Genesis/data/slates_binned/*` the same way
they are pointed at `runs_exp0026/*.pt`.

Why this exists
---------------
The pool diagnostics (`pool_inspect`, `pool_survey`) read a cache of
``{actions, dv_true, dv_pred per model}`` grouped by slate. The two existing
cache builders (`exp0026_selection_pressure.py`, `expB_multistep_eval.py`) both
reach the raw pile state through `PileSweepData`'s ``_{k}_data.pt`` layout,
which a binned corpus does not use -- it is written as ``step{k}.pt`` by
`Genesis/binned_slate_collection.py` and read by
`Genesis/binned_slate_dataset.py`. So this is the third builder, and it differs
from the other two only in where the states come from and in one deliberate
simplification: it EMBEDS ``occ0`` and the workspace bounds in the cache
instead of making the reader reload the raw dataset to recover them (see
`pool_common.load_occ0_for_slate`, which prefers the embedded copy when present).

Models scored
-------------
The three promoted instances in `weights/`, plus persistence as the mandatory
non-learned baseline:

    persistence        predict "nothing moves": dv = 0 for every candidate.
                       Degenerate AS A RANKER (see below) -- kept as the
                       documented prediction baseline only.
    random             uniform noise: the honest ranking floor, and the
                       baseline every ranking comparison here must beat
    nfd                MODEL-0003, the 3-action-channel NFD UNet, image space
    visual-switched    MODEL-0001, 6 per-push-length-bin ridge operators on
                       32x32 canonical push-frame occupancy, image space
    descriptor         MODEL-0002, 94-dim analytic descriptors, scored through
                       its point-mass value readout (it has no decoder and
                       cannot produce an image at all)

Each model's prediction pipeline is IMPORTED from the code that was validated
against it, not reimplemented: `multistep-rollout/rollout.py`'s image-space
step functions for the first two, `desc-mlp/eval_control.py`'s descriptor
pipeline (the one EXP-0012 ran on this very corpus) for the third.

`dv` convention
---------------
``dv = value(after) - value(before)``, a COST: negative = the push improved the
goal, and lower is better. This matches every ranking metric in
`experiments/METRICS.md`. Within one slate ``value(before)`` is a constant, so
subtracting it changes no ranking, no ``slateN`` and no ``|R_K|`` -- it only
recentres the histogram on "did this push help or hurt", which is what a reader
of one pool wants to see.

The descriptor model's number is not on this scale at all (it is a distance
field sampled at a predicted point mass, not an integrated value), so only its
ORDER is meaningful. Every panel that consumes `dv_pred` uses it through
`argsort`/`argmin` only; the printed "predicted dv" column is the one place the
raw number appears, and it is not comparable across models.

Usage
-----
    PYTHONPATH=. python -u scripts/probes/binned_pool_cache.py \\
        Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm \\
        --out experiments/temp/binned-pools/dv_cache.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
for extra in ("", "experiments/temp/dmdc-lenbins", "experiments/temp/stage3-slaten",
              "experiments/temp/desc-mlp"):
    sys.path.insert(0, str(REPO / extra))

from control_utility_test import lyapunov, lyapunov_weights          # noqa: E402
from transforms.functional import particles_to_occupancy             # noqa: E402
from fit_linear_foresight import actions_to_pixels                   # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus           # noqa: E402
from utils import git_provenance                                     # noqa: E402

# Identical to rollout.py / eval_control.py so occupancies are comparable with
# every earlier number on these models.
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def occ_of(states):
    return particles_to_occupancy(states[..., :3].to(DEVICE), BOUNDS, (GRID, GRID),
                                  footprint_radius=RADIUS)


def chunked(fn, n, size=512):
    """Apply ``fn(lo, hi)`` over row blocks and concatenate -- the pool is
    20k rows and the NFD forward will not fit in one batch."""
    return torch.cat([fn(lo, min(lo + size, n)) for lo in range(0, n, size)])


def build(corpus_dir, goal_shape="corner", step=0):
    corpus = BinnedSlateCorpus.load(corpus_dir)
    rows = corpus.step(step)
    n = len(rows)
    print(f"corpus: {corpus.n_slates} slates x {corpus.n_actions} candidates "
          f"({corpus.spawn_style} spawn, {n} rows at step {step})", flush=True)

    states, states_ = rows.states.float(), rows.states_.float()
    p_start, p_stop = rows.p_starts[:, :2].float(), rows.p_stops[:, :2].float()
    angles = rows.angles.float()
    actions = torch.cat([p_start, p_stop], dim=1)                 # (n, 4) metres
    s_px, e_px = actions_to_pixels(actions, WS_MIN, WS_MAX, (GRID, GRID))
    length_m = (p_stop - p_start).norm(dim=-1)

    occ0 = chunked(lambda a, b: occ_of(states[a:b]), n)
    occ1 = chunked(lambda a, b: occ_of(states_[a:b]), n)

    dw = lyapunov_weights((GRID, GRID), goal_shape, DEVICE)

    v_before = lyapunov(occ0, dw).cpu()
    dv_true = (lyapunov(occ1, dw).cpu() - v_before)

    # persistence predicts no change at all, so its dv_pred is 0 for EVERY
    # candidate: as a ranker it is degenerate (argmin always returns index 0
    # and the induced order is row order, not a prediction). It is kept because
    # it is the documented prediction baseline, but `random` below is the
    # honest ranking floor and is the one to read against.
    preds = {"persistence": torch.zeros(n)}
    preds["random"] = torch.rand(n, generator=torch.Generator().manual_seed(0))

    # --- MODEL-0003: NFD, 3 action channels ---------------------------------
    sys.path.insert(0, str(REPO / "experiments/temp/multistep-rollout"))
    from rollout import make_nfd_step, make_linear_step, RawStub      # noqa: F401
    nfd = _nfd_step(str(REPO / "weights/MODEL-0003-nfd-multistep-finetuned/checkpoint.pth"))
    occ_nfd = chunked(lambda a, b: nfd(occ0[a:b], p_start[a:b], p_stop[a:b], angles[a:b]), n)
    preds["nfd"] = (lyapunov(occ_nfd, dw).cpu() - v_before)
    del occ_nfd

    # --- MODEL-0001: switched-linear visual operator ------------------------
    ckpt = torch.load(REPO / "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                      map_location=DEVICE, weights_only=False)
    lin_step, _edges = make_linear_step(ckpt, desc_dim=0, switched=True)
    occ_lin = chunked(lambda a, b: lin_step(occ0[a:b], s_px[a:b].to(DEVICE),
                                            e_px[a:b].to(DEVICE),
                                            length_m[a:b].to(DEVICE)), n)
    preds["visual-switched"] = (lyapunov(occ_lin, dw).cpu() - v_before)
    del occ_lin

    # --- MODEL-0002: descriptor-only, point-mass value readout --------------
    from eval_control import (compute_descriptors, predict_switched, load_switched,
                              com_world_pixel, bilinear_sample)
    sw_ops, sw_edges, _slices = load_switched()
    _o0, _o1, phi0, len_d, sp_d, ep_d = compute_descriptors(states, states_, p_start, p_stop)
    phi_pred = predict_switched(sw_ops, sw_edges, phi0, len_d)
    com = phi_pred[:, 3:5]
    wr, wc = com_world_pixel(com[:, 0], com[:, 1], sp_d, ep_d, GRID, GRID)
    # Not on dv's scale (a sampled distance field, not an integrated value);
    # only its ORDER is used. See the module docstring.
    preds["descriptor"] = bilinear_sample(dw.cpu(), wr.cpu(), wc.cpu())

    # occ0 is shared by every candidate of a slate, so one image per slate is
    # all the figure needs -- and storing it here is what frees the reader from
    # reloading the raw corpus through a loader that cannot read this layout.
    first_of_slate = [int((rows.slate_idx == s).nonzero()[0]) for s in range(corpus.n_slates)]
    return {
        "ep": rows.slate_idx.long(),
        "actions": actions,
        "angles": angles,
        "len_realized": rows.len_realized,
        "bin_realized": rows.bin_realized,
        "occ0": occ0[first_of_slate].cpu(),          # (n_slates, GRID, GRID)
        "ws_min": WS_MIN, "ws_max": WS_MAX,
        "dv": {goal_shape: {"dv_true": dv_true, **preds}},
        "config": {"corpus": str(corpus_dir), "step": step, "goal": goal_shape,
                   "grid": GRID, "bounds": BOUNDS, "spawn_style": corpus.spawn_style,
                   "value_fn": "lyapunov", "dv": "value(after) - value(before)",
                   "models": {"nfd": "MODEL-0003", "visual-switched": "MODEL-0001",
                              "descriptor": "MODEL-0002"}},
        "provenance": git_provenance(),
    }


def _nfd_step(ckpt_path):
    """MODEL-0003's forward, built the same way `rollout.make_nfd_step` builds
    the base 3-channel NFD -- same plate rasterisation, same channel order --
    but pointed at the promoted fine-tuned checkpoint rather than
    `Baselines/NFD/runs/nfd_3ch/unet_best.pth`."""
    from Baselines.NFD.predictor import NFDPredictor, _plate_geometry_px
    from transforms.functional import draw_plate_soft
    from rollout import RawStub

    pred = NFDPredictor(ckpt_path, channels=3, name="nfd")
    pred.model.to(DEVICE).eval()
    raw = RawStub()
    plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)
    ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(DEVICE)[:2]

    def step_fn(occ_in, p_start_xy, p_stop_xy, angle):
        start_px = p_start_xy.to(DEVICE) * raw.to_pxl + ctr_xy
        stop_px = p_stop_xy.to(DEVICE) * raw.to_pxl + ctr_xy
        H, W = occ_in.shape[-2:]
        r_start = draw_plate_soft(start_px, angle.to(DEVICE), (H, W), plate_x_px,
                                  plate_y_px, intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle.to(DEVICE), (H, W), plate_x_px,
                                 plate_y_px, intensity=1.0, sigma=sigma)
        x = torch.stack([occ_in.to(DEVICE), r_start, r_stop], dim=1)
        with torch.no_grad():
            return torch.sigmoid(pred.model(x))[:, 0]

    return step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = build(args.corpus, args.goal, args.step)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)

    g = cache["dv"][args.goal]
    print(f"\ndv_true: mean {float(g['dv_true'].mean()):+.5f}  "
          f"sd {float(g['dv_true'].std()):.5f}  "
          f"frac improving {float((g['dv_true'] < 0).float().mean()):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
