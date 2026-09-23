"""EXP-0021 -- 9-cell dV cache builder for DS-0001 (`slates_binned`).

Extends `scripts/probes/binned_pool_cache.py` (1 cell: `corner` x `lyapunov`)
to the 9 cells this experiment pre-registered:

    goals       corner, ring_O (letter 'O'), letter_X (letter 'X')
    value fns   lyapunov, mass_in_region, signed_mass_in_region

and swaps the arms onto the SHARED-SPLIT checkpoints that EXP-0021's
`provenance.split` requires. `scripts/probes/*` is NOT modified; the pieces
that are reusable are imported from there / from the code that was validated
against each model.

SENSE CONVENTION -- read this before touching a number
------------------------------------------------------
`scripts/probes/pool_survey.py` is hard-wired to `argmin(dv_pred)` and
`slate_n_capture(..., higher_is_better=False)`. So **everything stored under
`cache["dv"][cell]` is a COST: LOWER IS BETTER, for `dv_true` and for every
arm alike.** The three value functions do not natively agree on that, so each
carries an explicit sign:

    lyapunov              COST  (lower better)   sign = +1
    mass_in_region        VALUE (higher better)  sign = -1
    signed_mass_in_region VALUE (higher better)  sign = -1

    dv_true = sign * ( value(occ_after) - value(occ_before) )

For the VALUE functions this means the stored `dv_true` is the NEGATED value
gain: a push that puts more mass in the region has a more negative `dv_true`.
`value(occ_before)` is constant within a slate, so subtracting it changes no
ranking, no `slateN` and no `|R_K|`.

The same `sign` is applied to every arm's score, so an arm's number is always
on the cost side too. The `descriptor` arm is the exception in SCALE (its
point-mass readout is a field sampled at a predicted centre of mass, not an
integrated value) but not in SENSE -- only its ORDER is meaningful, and that
order is a cost order like everyone else's.

Arms
----
    random            uniform noise, seeded. THE RANKING FLOOR.
    persistence       dv = 0 for every candidate. DEGENERATE AS A RANKER
                      (argmin returns row 0, so the induced order is row
                      order, not a prediction). Reference row only -- it is
                      flagged `degenerate_rankers` in the cache config.
    descriptor        the CLEAN shared-split refit of MODEL-0002's recipe,
                      `experiments/temp/exp0021-shared-split/
                      desc94_switched_sharedsplit.pt`. MODEL-0002 itself is
                      NOT used: it was fit on a legacy 170-file split that
                      leaks 17 of the 21 shared-split test files.
    visual-switched   `Baselines/LinearForesight/runs/operators_res32.pt`
                      (same recipe as MODEL-0001, correct split).
    nfd               `Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`
                      (randlen-trained; MODEL-0003 is fit on
                      `slates_multistep` and is excluded by design).

MODEL-0001 / MODEL-0002 / MODEL-0003 are all excluded -- see the record's
`provenance.split`.

Extension point for the latent arms
-----------------------------------
Arms are a registry, not a hard-coded block. A new arm is one call:

    register_occ_arm("lejepa-decoder", lambda ctx: ...)        # (n,H,W) occ
    register_score_arm("lejepa-valuehead", lambda ctx, cell: ...)  # (n,) cost

`ctx` (see `Ctx`) carries occ0, states, the action tensors, push-frame pixel
endpoints, push length and the slate ids; `cell` carries the goal mask, the
distance field, the value fn and its sign. An occ-arm is scored through the
IDENTICAL value path as the image arms. Nothing is stubbed: an arm that is
not registered simply does not appear in the cache.

Usage
-----
    # smoke test, 3 slates
    PYTHONPATH=. python -u \\
      experiments/EXP-0021-five-way-ranking-comparison/code/build_cache_9cell.py \\
      Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm \\
      --max-slates 3 --out experiments/temp/exp0021-scorer/dv_cache_smoke.pt

    # full build (all 20 slates x 1000 candidates x 9 cells)
    PYTHONPATH=. python -u \\
      experiments/EXP-0021-five-way-ranking-comparison/code/build_cache_9cell.py \\
      Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm \\
      --out experiments/temp/exp0021-scorer/dv_cache_9cell.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
for extra in ("", "experiments/temp/dmdc-lenbins", "experiments/temp/desc-mlp",
              "experiments/temp/multistep-rollout"):
    p = str(REPO / extra) if extra else str(REPO)
    if p not in sys.path:
        sys.path.insert(0, p)

from control_utility_test import lyapunov, lyapunov_weights            # noqa: E402
from transforms.functional import particles_to_occupancy               # noqa: E402
from fit_linear_foresight import actions_to_pixels                     # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus             # noqa: E402
from utils import git_provenance                                       # noqa: E402
from Baselines.common.goals import (                                   # noqa: E402
    letter_mask, dist_field_from_mask, mass_in_region, signed_mass_in_region,
    slate_n_capture,
)

# Identical to binned_pool_cache.py / rollout.py / eval_control.py so every
# occupancy is comparable with every earlier number on these models.
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])

DESC_CKPT = REPO / "experiments/temp/exp0021-shared-split/desc94_switched_sharedsplit.pt"
VISUAL_CKPT = REPO / "Baselines/LinearForesight/runs/operators_res32.pt"
NFD_CKPT = REPO / "Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth"

GOALS = ["corner", "ring_O", "letter_X"]
# name -> (callable(occ, mask, dw) -> (n,), sign, higher_is_better_natively)
VALUE_FNS = {
    "lyapunov": (lambda occ, mask, dw: lyapunov(occ, dw), +1.0, False),
    "mass_in_region": (lambda occ, mask, dw: mass_in_region(occ, mask), -1.0, True),
    "signed_mass_in_region": (lambda occ, mask, dw: signed_mass_in_region(occ, mask),
                              -1.0, True),
}
DEGENERATE_RANKERS = ["persistence"]


# --------------------------------------------------------------------------
# goals
# --------------------------------------------------------------------------
def build_goal(name: str, H: int, W: int, device):
    """Return (mask bool (H,W) numpy, dw float32 torch (H,W)) in convention A
    (row = world x, col = world y; see `Baselines/common/goals.py`).

    `corner` is cross-checked against `control_utility_test.lyapunov_weights`
    -- the two must agree, and a mismatch is reported, not silently patched.
    """
    if name == "corner":
        mask = np.zeros((H, W), dtype=bool)
        mask[: H // 2, : W // 2] = True          # lyapunov_weights's own corner
    elif name == "ring_O":
        mask = letter_mask("O", H, W)
    elif name == "letter_X":
        mask = letter_mask("X", H, W)
    else:
        raise ValueError(name)
    dw = torch.from_numpy(dist_field_from_mask(mask)).to(device)
    if name == "corner":
        ref = lyapunov_weights((H, W), "corner", device)
        gap = float((dw - ref).abs().max())
        if gap > 1e-6:
            print(f"  [WARN] corner dist field disagrees with "
                  f"lyapunov_weights('corner'): max |diff| = {gap:.3e}")
        else:
            print("  [ok] corner: dist_field_from_mask == lyapunov_weights('corner') "
                  "(max |diff| = 0)")
    return mask, dw


# --------------------------------------------------------------------------
# arm registry -- the extension point for the latent arms
# --------------------------------------------------------------------------
@dataclass
class Ctx:
    """Everything an arm may need about the pool, built once."""
    occ0: torch.Tensor          # (n,H,W) on `device`
    states: torch.Tensor        # (n,N,>=3) cpu
    p_start: torch.Tensor       # (n,2) metres, cpu
    p_stop: torch.Tensor        # (n,2) metres, cpu
    angles: torch.Tensor        # (n,) cpu
    s_px: torch.Tensor          # (n,2) push-frame pixel start
    e_px: torch.Tensor          # (n,2)
    length_m: torch.Tensor      # (n,) cpu
    slate_idx: torch.Tensor     # (n,) long cpu
    device: str
    n: int
    cache: dict = field(default_factory=dict)   # arms may memoise here


@dataclass
class Cell:
    goal: str
    value_fn: str
    mask: torch.Tensor          # (H,W) bool on `device`
    dw: torch.Tensor            # (H,W) on `device`
    sign: float                 # +1 cost-native, -1 value-native
    fn: object

    @property
    def name(self):
        return f"{self.goal}__{self.value_fn}"

    def value(self, occ):
        return self.fn(occ, self.mask, self.dw).cpu()

    def cost_delta(self, occ_after, v_before):
        return self.sign * (self.value(occ_after) - v_before)


OCC_ARMS: dict = {}     # name -> callable(Ctx) -> (n,H,W) predicted occupancy
SCORE_ARMS: dict = {}   # name -> callable(Ctx, Cell) -> (n,) COST (lower better)
BUILTIN_ARMS = ("random", "persistence", "descriptor", "visual-switched", "nfd")


def register_occ_arm(name, fn):
    """An arm that predicts an occupancy image. Scored through the IDENTICAL
    value path as every other image arm, for all 9 cells."""
    OCC_ARMS[name] = fn


def register_score_arm(name, fn):
    """An arm that produces a per-candidate COST directly (no image). Must
    already be sign-corrected: LOWER IS BETTER."""
    SCORE_ARMS[name] = fn


def chunked(fn, n, size=256):
    return torch.cat([fn(lo, min(lo + size, n)) for lo in range(0, n, size)])


# --------------------------------------------------------------------------
# built-in arms
# --------------------------------------------------------------------------
def _arm_random(ctx, cell, seed=0):
    # One fixed draw for the whole pool: it is noise, and reusing it across
    # cells keeps the floor comparable cell-to-cell.
    if "random" not in ctx.cache:
        ctx.cache["random"] = torch.rand(ctx.n,
                                         generator=torch.Generator().manual_seed(seed))
    return ctx.cache["random"].clone()


def _arm_persistence(ctx, cell):
    return torch.zeros(ctx.n)


def _make_visual_arm(device):
    """`operators_res32.pt` re-keyed onto `rollout.make_linear_step`'s expected
    layout (`switched_ops`/`global_op`). Same fit recipe as MODEL-0001, but the
    correct (shared) split."""
    import rollout
    # rollout.py hard-codes a module-level DEVICE = "cuda if available" and
    # pushes the operators onto it at construction. This box's GPU belongs to
    # another job, and mixing devices raises anyway -- pin it to ours.
    rollout.DEVICE = device
    from rollout import make_linear_step
    ck = torch.load(VISUAL_CKPT, map_location=device, weights_only=False)
    assert ck["res"] == 32 and abs(ck["crop"] - 1.0) < 1e-9, \
        f"make_linear_step hard-codes RES=32/crop=1.0, ckpt has {ck['res']}/{ck['crop']}"
    shim = {"bin_edges": ck["bin_edges"],
            "switched_ops": ck["operators"],
            "global_op": ck["single_operator"]}
    step_fn, _edges = make_linear_step(shim, desc_dim=0, switched=True)

    def occ_fn(ctx):
        return chunked(lambda a, b: step_fn(ctx.occ0[a:b],
                                            ctx.s_px[a:b].to(device),
                                            ctx.e_px[a:b].to(device),
                                            ctx.length_m[a:b].to(device)), ctx.n)
    return occ_fn


def _make_nfd_arm(device):
    """The randlen-trained 3-channel NFD UNet, built exactly as
    `rollout.make_nfd_step` builds the base NFD (same plate rasterisation,
    same channel order), pointed at the randlen checkpoint."""
    from Baselines.NFD.predictor import NFDPredictor, _plate_geometry_px
    from transforms.functional import draw_plate_soft
    from rollout import RawStub

    pred = NFDPredictor(str(NFD_CKPT), channels=3, name="nfd")
    pred.model.to(device).eval()
    raw = RawStub()
    plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)
    ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(device)[:2]

    def step(occ_in, p0, p1, ang):
        start_px = p0.to(device) * raw.to_pxl + ctr_xy
        stop_px = p1.to(device) * raw.to_pxl + ctr_xy
        H, W = occ_in.shape[-2:]
        r0 = draw_plate_soft(start_px, ang.to(device), (H, W), plate_x_px, plate_y_px,
                             intensity=1.0, sigma=sigma)
        r1 = draw_plate_soft(stop_px, ang.to(device), (H, W), plate_x_px, plate_y_px,
                             intensity=1.0, sigma=sigma)
        x = torch.stack([occ_in.to(device), r0, r1], dim=1)
        with torch.no_grad():
            return torch.sigmoid(pred.model(x))[:, 0]

    def occ_fn(ctx):
        return chunked(lambda a, b: step(ctx.occ0[a:b], ctx.p_start[a:b],
                                         ctx.p_stop[a:b], ctx.angles[a:b]), ctx.n)
    return occ_fn


def _make_descriptor_arm():
    """The clean shared-split 94-dim switched operator, read out through
    `desc-mlp/eval_control.py`'s point-mass readout (reused, not reinvented --
    it is the path EXP-0012 ran on this very corpus and it already covers all
    three value functions).

    It has no image, so its number is NOT on dv's scale; only its order is
    used. `desc_pointmass_value` returns a COST for `lyapunov` (a distance
    sampled at the predicted COM) and a VALUE for the two mass functions, so
    the cell's own `sign` is applied to put all three on the cost side --
    exactly the same sign table the ground truth uses.
    """
    from eval_control import (compute_descriptors, predict_switched,
                              com_world_pixel, desc_pointmass_value)

    ck = torch.load(DESC_CKPT, map_location="cpu", weights_only=False)
    ops, edges = ck["ops"], ck["bin_edges"]

    def prep(ctx):
        if "desc" not in ctx.cache:
            _o0, _o1, phi0, len_d, sp_d, ep_d = compute_descriptors(
                ctx.states[..., :3], ctx.states[..., :3], ctx.p_start, ctx.p_stop)
            phi_pred = predict_switched(ops, edges, phi0, len_d)
            com = phi_pred[:, 3:5]                       # SD["com"]
            mass_hat = phi_pred[:, 0] * (GRID * GRID)    # SD["global_mass"]
            wr, wc = com_world_pixel(com[:, 0], com[:, 1], sp_d, ep_d, GRID, GRID)
            ctx.cache["desc"] = (wr.cpu(), wc.cpu(), mass_hat.cpu())
        return ctx.cache["desc"]

    def score_fn(ctx, cell):
        wr, wc, mass_hat = prep(ctx)
        v = desc_pointmass_value(cell.value_fn, wr, wc, mass_hat,
                                 cell.dw.cpu(), cell.mask.cpu())
        return cell.sign * v.float()

    return score_fn


# --------------------------------------------------------------------------
# letter-mask overlay sanity check (commit 28271c09)
# --------------------------------------------------------------------------
def check_letter_overlay(mask, name, n_objects=20, seed=0):
    """Rasterise a LEGAL material configuration generated FROM the mask and
    confirm most of its occupancy mass lands inside that same mask. `corner`
    is transpose-invariant and cannot detect an axis flip; `ring_O` and
    `letter_X` can, and only became correct at commit 28271c09."""
    from Baselines.common.goal_configs import mask_to_configuration
    res = mask_to_configuration(mask, n_objects=n_objects, bounds=BOUNDS, seed=seed)
    poses = res.poses if hasattr(res, "poses") else res.config
    xyz = torch.as_tensor(np.asarray(poses)[:, :3], dtype=torch.float32)[None]
    occ = particles_to_occupancy(xyz, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)[0]
    m = torch.from_numpy(mask)
    inside = float((occ * m).sum())
    total = max(float(occ.sum()), 1e-9)
    frac = inside / total
    # A cube footprint is wider than a thin glyph stroke, so even a perfect
    # overlay spills; the discriminating comparison is against the TRANSPOSE.
    m_t = torch.from_numpy(np.ascontiguousarray(mask.T))
    frac_t = float((occ * m_t).sum()) / total
    return frac, frac_t


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(corpus_dir, step=0, max_slates=None, device="cpu", seed=0,
          arms=("random", "persistence", "descriptor", "visual-switched", "nfd")):
    corpus = BinnedSlateCorpus.load(corpus_dir)
    rows = corpus.step(step)
    slate_idx = rows.slate_idx.long()
    keep = torch.ones(len(slate_idx), dtype=torch.bool)
    slates = sorted(slate_idx.unique().tolist())
    if max_slates is not None:
        slates = slates[:max_slates]
        keep = torch.isin(slate_idx, torch.tensor(slates))
    sel = keep.nonzero(as_tuple=True)[0]
    n = len(sel)
    print(f"corpus: {corpus.n_slates} slates x {corpus.n_actions} candidates "
          f"({corpus.spawn_style} spawn); using {len(slates)} slates / {n} rows "
          f"at step {step}, device={device}", flush=True)

    states = rows.states.float()[sel]
    states_ = rows.states_.float()[sel]
    p_start = rows.p_starts[:, :2].float()[sel]
    p_stop = rows.p_stops[:, :2].float()[sel]
    angles = rows.angles.float()[sel]
    slate_idx = slate_idx[sel]
    actions = torch.cat([p_start, p_stop], dim=1)
    s_px, e_px = actions_to_pixels(actions, WS_MIN, WS_MAX, (GRID, GRID))
    length_m = (p_stop - p_start).norm(dim=-1)

    def occ_of(st):
        return particles_to_occupancy(st[..., :3].to(device), BOUNDS, (GRID, GRID),
                                      footprint_radius=RADIUS)

    occ0 = chunked(lambda a, b: occ_of(states[a:b]), n)
    occ1 = chunked(lambda a, b: occ_of(states_[a:b]), n)

    ctx = Ctx(occ0=occ0, states=states, p_start=p_start, p_stop=p_stop, angles=angles,
              s_px=s_px, e_px=e_px, length_m=length_m, slate_idx=slate_idx,
              device=device, n=n)

    # ---- register the arms this record carries ---------------------------
    # Only the BUILT-IN entries are reset, so an arm registered from outside
    # this module (the latent arms, when they exist) survives a build() call.
    for k in BUILTIN_ARMS:
        OCC_ARMS.pop(k, None)
        SCORE_ARMS.pop(k, None)
    if "random" in arms:
        register_score_arm("random", lambda c, cell: _arm_random(c, cell, seed))
    if "persistence" in arms:
        register_score_arm("persistence", _arm_persistence)
    if "descriptor" in arms:
        register_score_arm("descriptor", _make_descriptor_arm())
    if "visual-switched" in arms:
        register_occ_arm("visual-switched", _make_visual_arm(device))
    if "nfd" in arms:
        register_occ_arm("nfd", _make_nfd_arm(device))
    print(f"arms: occ={list(OCC_ARMS)} score={list(SCORE_ARMS)}", flush=True)

    # ---- goals + overlay sanity ------------------------------------------
    print("\n=== goals ===", flush=True)
    goal_objs, overlay = {}, {}
    for g in GOALS:
        mask_np, dw = build_goal(g, GRID, GRID, device)
        goal_objs[g] = (mask_np, dw)
        f, ft = check_letter_overlay(mask_np, g, seed=seed)
        overlay[g] = {"mass_frac_in_own_mask": f, "mass_frac_in_transposed_mask": ft}
        flag = "" if f >= ft else "   <-- TRANSPOSED?!"
        print(f"  {g:10s} mask area={mask_np.mean():.3f}  goal-config mass inside "
              f"own mask={f:.3f}  inside transpose={ft:.3f}{flag}", flush=True)

    # ---- predicted occupancies (once, shared by all 9 cells) -------------
    occ_pred = {}
    for name, fn in OCC_ARMS.items():
        occ_pred[name] = fn(ctx)
        print(f"  [occ] {name}: {tuple(occ_pred[name].shape)}", flush=True)

    # ---- the 9 cells ------------------------------------------------------
    dv, diag, asserts = {}, {}, {}
    for g in GOALS:
        mask_np, dw = goal_objs[g]
        mask = torch.from_numpy(mask_np).to(device)
        for vf, (fn, sign, hib) in VALUE_FNS.items():
            cell = Cell(goal=g, value_fn=vf, mask=mask, dw=dw, sign=sign, fn=fn)
            v0 = cell.value(occ0)
            v1 = cell.value(occ1)
            dv_true = sign * (v1 - v0)
            entry = {"dv_true": dv_true}
            for name in OCC_ARMS:
                entry[name] = cell.cost_delta(occ_pred[name], v0)
            for name, sfn in SCORE_ARMS.items():
                entry[name] = sfn(ctx, cell)
            dv[cell.name] = entry
            diag[cell.name] = _degeneracy(dv_true, slate_idx, slates)
            asserts[cell.name] = _sense_assertions(dv_true, v1, slate_idx, slates,
                                                   hib, entry["random"])
            d, a = diag[cell.name], asserts[cell.name]
            print(f"  [{cell.name:34s}] sign={sign:+.0f} "
                  f"frac(dv_true==0)={d['frac_zero']:.4f} "
                  f"spread(max-min) med={d['spread_median']:.5f} "
                  f"| oracle_ok={a['oracle_is_true_best']} "
                  f"slateN oracle={a['slateN_oracle']:.3f} "
                  f"random={a['slateN_random']:+.3f}", flush=True)

    first_of_slate = [int((slate_idx == s).nonzero()[0]) for s in slates]
    cache = {
        "ep": slate_idx,
        "actions": actions,
        "angles": angles,
        "len_realized": rows.len_realized[sel],
        "bin_realized": rows.bin_realized[sel],
        "occ0": occ0[first_of_slate].cpu(),
        "ws_min": WS_MIN, "ws_max": WS_MAX,
        "dv": dv,
        "config": {
            "corpus": str(corpus_dir), "step": step, "grid": GRID, "bounds": BOUNDS,
            "spawn_style": corpus.spawn_style, "slates": slates, "seed": seed,
            "device": device,
            "cells": list(dv.keys()), "goals": GOALS,
            "value_fn_sign": {k: v[1] for k, v in VALUE_FNS.items()},
            "dv": ("sign * (value(after) - value(before)); sign=+1 for the COST "
                   "lyapunov and -1 for the VALUE mass fns, so dv is ALWAYS a "
                   "cost (lower better) -- what pool_survey.py's argmin and "
                   "slate_n_capture(higher_is_better=False) require"),
            "degenerate_rankers": DEGENERATE_RANKERS,
            "models": {"descriptor": str(DESC_CKPT.relative_to(REPO)),
                       "visual-switched": str(VISUAL_CKPT.relative_to(REPO)),
                       "nfd": str(NFD_CKPT.relative_to(REPO))},
            "excluded_models": ["weights/MODEL-0001-stage2-visual-switched",
                                "weights/MODEL-0002-descriptor-only-D-all-local",
                                "weights/MODEL-0003-nfd-multistep-finetuned"],
            "goal_overlay_check": overlay,
            "plugin_arms": [k for k in list(OCC_ARMS) + list(SCORE_ARMS)
                            if k not in BUILTIN_ARMS],
        },
        "diagnostics": diag,
        "assertions": asserts,
        "provenance": git_provenance(),
    }
    return cache


def _degeneracy(dv_true, slate_idx, slates):
    spreads, sds = [], []
    for s in slates:
        t = dv_true[slate_idx == s]
        spreads.append(float(t.max() - t.min()))
        sds.append(float(t.std()))
    return {"frac_zero": float((dv_true == 0).float().mean()),
            "spread_median": float(np.median(spreads)),
            "spread_mean": float(np.mean(spreads)),
            "spread_min": float(np.min(spreads)),
            "per_slate_sd_mean": float(np.mean(sds)),
            "dv_true_sd": float(dv_true.std())}


def _sense_assertions(dv_true, v_after, slate_idx, slates, higher_is_better, rnd):
    """The proof the sense table is wired correctly.

    (1) argmin(dv_true) -- the cache's cost oracle -- must land on the
        candidate with the genuinely best TRUE value (max if the value fn is
        higher-is-better, min if it is a cost).
    (2) slate_n_capture(dv_true, dv_true, higher_is_better=False), i.e. the
        oracle scored the way pool_survey scores it, must be 1.
    (3) the seeded random floor must score ~0.
    """
    ok, cap_o, cap_r = [], [], []
    for s in slates:
        m = slate_idx == s
        t, va, r = dv_true[m], v_after[m], rnd[m]
        i = int(torch.argmin(t))
        j = int(torch.argmax(va) if higher_is_better else torch.argmin(va))
        ok.append(abs(float(va[i]) - float(va[j])) < 1e-9)
        c = slate_n_capture(t, t, higher_is_better=False)
        if c == c:
            cap_o.append(c)
        c = slate_n_capture(r, t, higher_is_better=False)
        if c == c:
            cap_r.append(c)
    return {"oracle_is_true_best": bool(np.all(ok)),
            "n_slates_checked": len(slates),
            "slateN_oracle": float(np.mean(cap_o)) if cap_o else float("nan"),
            "slateN_random": float(np.mean(cap_r)) if cap_r else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--max-slates", type=int, default=None,
                    help="smoke-test switch: use only the first N slates")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default -- a GPU training job owns this box) or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms",
                    default="random,persistence,descriptor,visual-switched,nfd")
    ap.add_argument("--plugin", action="append", default=[],
                    help="import path of a module exposing register(builder, device); "
                         "called BEFORE build() so its arms are in the registry. "
                         "`build()` only resets BUILTIN_ARMS, so plugin arms survive. "
                         "Use `path/to/mod.py` or a dotted module name.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for spec in args.plugin:
        if spec.endswith(".py"):
            import importlib.util
            mp = Path(spec).resolve()
            spec_o = importlib.util.spec_from_file_location(mp.stem, mp)
            mod = importlib.util.module_from_spec(spec_o)
            sys.modules[mp.stem] = mod
            spec_o.loader.exec_module(mod)
        else:
            import importlib
            mod = importlib.import_module(spec)
        mod.register(sys.modules[__name__], device=args.device)

    cache = build(args.corpus, step=args.step, max_slates=args.max_slates,
                  device=args.device, seed=args.seed,
                  arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    json.dump({"diagnostics": cache["diagnostics"], "assertions": cache["assertions"],
               "goal_overlay_check": cache["config"]["goal_overlay_check"],
               "provenance": cache["provenance"]},
              open(str(Path(args.out).with_suffix(".json")), "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
