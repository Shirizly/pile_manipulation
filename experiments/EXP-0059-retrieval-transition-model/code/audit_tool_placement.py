"""Tool-placement LEGALITY audit (2026-09-28, coordinator follow-up A).

The user noticed, in the retrieval visual-debug figures (`retrieval_debug.py`,
e.g. `q0908`), a query cube sitting almost exactly where the blade touches
down (push-frame u in roughly [-5, 0] mm, inside the blade's lateral span) --
i.e. the tool descending onto/through a cube at `p_start`, which is illegal in
the spirit of "pile-aware"/"placement-aware" sampling (the whole point of
those samplers is that the tool must never be placed on a cube). This script
tests EVERY recorded transition's touchdown pose against every cube's actual
rotated-square footprint with an exact SAT test
(`Baselines/common/cube_overlap.py::overlaps_rect_pairs`, extended in this
task to rectangles of different sizes so the thin blade can be tested against
a cube directly, rather than approximating both as same-size squares).

ROOT CAUSE (read first, before adding a filter and moving on -- see
`experiments/OPEN_ISSUES.md` for the full writeup):

  Every dataset audited here (DS-0008/9/10/11/12/13) is `n=20` single-layer,
  narrow-domain, produced with `pile_aware=True` action sampling (DS-0010's
  provenance is older/mixed and not fully reconstructable -- see below), i.e.
  `Genesis/sandbox_manipulation_clean.py::_apply_pile_aware_starts` +
  `_pile_aware_stops`. `_apply_pile_aware_starts`/`action_sampling.py::
  pile_contact_starts` computes a genuinely collision-free touchdown ("one
  particle-width behind the pile's near face"). But `_pile_aware_stops`
  (`Genesis/sandbox_manipulation_clean.py` ~line 1953-1966) THEN
  unconditionally clamps that start into the yaw-dependent sampling BOX
  (`action_sampling.sampling_box`, a pure tray-wall/blade-footprint bound with
  ZERO knowledge of where cubes are), because "the pile spreads well past its
  spawn extent" (that function's own comment: particle radius reaches p95
  34.6 mm / max 54 mm against a blade box of only 23.5-42.5 mm half-extent).
  Its own measurement, quoted verbatim in that function: **35.8% of pile-aware
  starts fall outside the box and get clamped** -- and the clamp trades
  "starts exactly at the pile edge" for "starts just inside the pile", which
  the author flagged as still sweeping material, but did NOT check lands
  clear of any INDIVIDUAL cube's footprint. That is the mechanism this script
  measures directly. DS-0010 draws from `overnight_randlen_train/{mixed,
  piled}_n20` + `Sean/n20` (see its `extract.py`) -- "piled" strongly implies
  the same `pile_aware` code path and thus the same clamp; the "mixed"/Sean
  portions' exact sampler flags are not recoverable from provenance alone, so
  DS-0010's own measured rate is the only evidence for it.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/audit_tool_placement.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from Baselines.common.cube_overlap import overlaps_rect_pairs  # noqa: E402
from model.retrieval.frame import yaw_from_quat  # noqa: E402

D = REPO / "Genesis/data/narrow_l20_n20"
RES = REPO / "experiments/EXP-0059-retrieval-transition-model/results/tool_placement_audit.json"

TOOL_LENGTH, TOOL_WIDTH = 0.04, 0.002     # Genesis/configs/basic.yaml plate.size[:2]
CUBE_SIZE = 0.005
HALF_BLADE = np.array([TOOL_LENGTH / 2, TOOL_WIDTH / 2])
HALF_CUBE = np.array([CUBE_SIZE / 2, CUBE_SIZE / 2])
MARGIN = 0.001  # 1 mm -- the "small-margin" variant

# (name, glob pattern, kind in {"chains", "pools", "flat"})
DATASETS = [
    ("DS-0008_train",       str(D / "train/_*_data.pt"),          "chains"),
    ("DS-0009_test_chains", str(D / "test_chains/_*_data.pt"),    "chains"),
    ("DS-0009_test_pools",  str(D / "test_pools/pools_*.pt"),     "pools"),
    ("DS-0010_extra_18_22", str(D / "extra_18_22/*_data.pt"),     "flat"),
    ("DS-0011_val_pools",   str(D / "val_pools/pools_*.pt"),      "pools"),
    ("DS-0012_reservoir",   str(D / "reservoir_dsC/_*_data.pt"),  "chains"),
    ("DS-0013_seqpools",    str(D / "seqpools_dsB/_*_data.pt"),   "chains"),
]


def _row_illegal(states0: torch.Tensor, p_start: torch.Tensor, angle: torch.Tensor, tol: float):
    """states0: (N,20,7), p_start: (N,2), angle: (N,) -> (N,) bool, (N,20) bool per-cube."""
    N, n, _ = states0.shape
    cube_xy = states0[:, :, :2].numpy().reshape(N * n, 2)
    cube_yaw = yaw_from_quat(states0[:, :, 3:7]).numpy().reshape(N * n)
    blade_xy = np.repeat(p_start.numpy(), n, axis=0)
    blade_yaw = np.repeat(angle.numpy(), n, axis=0)
    ov = overlaps_rect_pairs(blade_xy, blade_yaw, HALF_BLADE, cube_xy, cube_yaw, HALF_CUBE, tol=tol)
    ov = ov.reshape(N, n)
    return ov.any(axis=1), ov


def audit_file(path: str, kind: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    states0 = d["states"].float()
    N = states0.shape[0]
    p_start = d["p_starts"].float()[:, :2]
    angle = d["angles"].float()
    valid = d["valid"].bool().numpy() if "valid" in d else np.ones(N, dtype=bool)
    chain_step = d["chain_step"].numpy() if "chain_step" in d else None
    pool_idx = d["pool_idx"].numpy() if "pool_idx" in d else None
    start_kind = np.asarray(d["start_kind"]) if "start_kind" in d else np.array(["?"] * N)
    if "start_kind" in d and len(start_kind) != N:
        # pools: one start_kind per POOL (`pool_idx` unique count), not per row --
        # broadcast through pool_idx (see test_pools/val_pools schema).
        assert pool_idx is not None and len(start_kind) == len(np.unique(pool_idx)), \
            f"start_kind length {len(start_kind)} matches neither N={N} nor n_pools"
        start_kind = start_kind[pool_idx]

    illegal_0, per_cube_0 = _row_illegal(states0, p_start, angle, tol=0.0)
    illegal_margin, per_cube_margin = _row_illegal(states0, p_start, angle, tol=-MARGIN)

    flags = dict(illegal_0mm=torch.from_numpy(illegal_0),
                illegal_1mm_margin=torch.from_numpy(illegal_margin),
                n_illegal_cubes_0mm=torch.from_numpy(per_cube_0.sum(axis=1).astype(np.int64)),
                n_illegal_cubes_1mm_margin=torch.from_numpy(per_cube_margin.sum(axis=1).astype(np.int64)),
                source_row=torch.arange(N))
    out_path = Path(path).with_name(Path(path).stem + "_legality.pt")
    tmp = str(out_path) + ".tmp"
    torch.save(flags, tmp)
    import os
    os.replace(tmp, out_path)

    return dict(N=N, illegal_0=illegal_0, illegal_margin=illegal_margin,
               valid=valid, start_kind=start_kind, chain_step=chain_step,
               pool_idx=pool_idx, out_path=str(out_path))


def _rate(mask, denom_mask=None):
    if denom_mask is not None:
        mask = mask[denom_mask]
    n = len(mask)
    return dict(n=int(n), n_illegal=int(mask.sum()), frac=float(mask.mean()) if n else float("nan"))


def audit_dataset(name, pattern, kind):
    files = sorted(f for f in glob.glob(pattern) if "_legality" not in f)
    if not files:
        print(f"  {name}: NO FILES matched {pattern}")
        return None
    all_illegal_0, all_illegal_margin, all_valid, all_kind, all_step = [], [], [], [], []
    for f in files:
        r = audit_file(f, kind)
        all_illegal_0.append(r["illegal_0"]); all_illegal_margin.append(r["illegal_margin"])
        all_valid.append(r["valid"]); all_kind.append(r["start_kind"])
        all_step.append(r["chain_step"] if r["chain_step"] is not None else np.full(r["N"], -1))
        print(f"    wrote {r['out_path']}  N={r['N']}  illegal_0mm={int(r['illegal_0'].sum())}  "
             f"illegal_1mm={int(r['illegal_margin'].sum())}")
    illegal_0 = np.concatenate(all_illegal_0)
    illegal_margin = np.concatenate(all_illegal_margin)
    valid = np.concatenate(all_valid)
    kinds = np.concatenate(all_kind)
    steps = np.concatenate(all_step)

    summary = dict(n_files=len(files),
                  all_rows=_rate(illegal_0), all_rows_1mm=_rate(illegal_margin),
                  valid_rows=_rate(illegal_0, valid), valid_rows_1mm=_rate(illegal_margin, valid))
    summary["by_start_kind"] = {
        k: dict(all=_rate(illegal_0[kinds == k]), valid_only=_rate(illegal_0[(kinds == k) & valid]))
        for k in sorted(set(kinds.tolist())) if k != "?"
    }
    if (steps >= 0).any():
        summary["by_chain_step"] = {
            int(s): dict(all=_rate(illegal_0[steps == s]), valid_only=_rate(illegal_0[(steps == s) & valid]))
            for s in sorted(set(steps[steps >= 0].tolist()))
        }
    return summary


def main():
    out = {}
    for name, pattern, kind in DATASETS:
        print(f"auditing {name} ...")
        s = audit_dataset(name, pattern, kind)
        if s is not None:
            out[name] = s
            print(f"  {name}: {s['valid_rows']['n_illegal']}/{s['valid_rows']['n']} "
                 f"valid rows illegal (0mm) = {s['valid_rows']['frac']:.4f}  "
                 f"({s['valid_rows_1mm']['frac']:.4f} at 1mm margin)")

    # DS-0009: do pools cluster illegal rows more than chains?
    if "DS-0009_test_pools" in out and "DS-0009_test_chains" in out:
        out["DS-0009_pools_vs_chains"] = dict(
            pools_frac=out["DS-0009_test_pools"]["valid_rows"]["frac"],
            chains_frac=out["DS-0009_test_chains"]["valid_rows"]["frac"])

    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(out, indent=1))
    print("wrote", RES)


if __name__ == "__main__":
    main()
