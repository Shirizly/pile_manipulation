"""Shared loading/geometry utilities for pool_inspect.py and pool_survey.py.

Both read a dV cache (``runs_exp0026/*.pt``, ``runs_expB/*.pt``, produced by
``scripts/probes/exp0026_selection_pressure.py`` / ``expB_multistep_eval.py``)
and need the same three things the cache itself does not directly hand over:

1. the set of candidate rows belonging to one slate (``cache["ep"]``);
2. the K-weighted M_k/P_k/R_k curve in RAW Lyapunov units (docs/experiments/
   METRICS.md defines the normalised ``slateK_exact`` from the same M_k/P_k;
   this module exposes the un-normalised pair so a reader can see the actual
   value at stake, not just the fraction captured);
3. the step-0 occupancy image a slate's candidates were drawn from, which the
   cache does not store at all -- only actions and true/predicted dV -- so it
   has to be reloaded from the same raw dataset the cache's own config points
   at.

``w_r(K)`` is imported from ``exp0026_kcurve_exact``, already validated there
against METRICS.md's published EXP-0026 numbers (``--self-test``), rather than
reimplemented -- one combinatorial-weight implementation, not two that could
silently disagree.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.probes.exp0026_kcurve_exact import _w

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def admissible_slates(cache, goal, min_slate=8):
    """Slate ids with >= min_slate candidates and nonzero true-dV spread --
    the same admission rule ``exp0026_kcurve_exact.sweep`` uses, so "every
    slate" means the identical population every other script here scores."""
    ep = cache["ep"]
    dv_true = cache["dv"][goal]["dv_true"]
    out = []
    for e in ep.unique().tolist():
        sel = (ep == e).nonzero(as_tuple=True)[0]
        if sel.numel() < min_slate:
            continue
        if float(dv_true[sel].std()) < 1e-9:
            continue
        out.append(e)
    return out


def slate_rows(cache, goal, slate_id):
    """(row indices, goal dict) for one slate."""
    ep = cache["ep"]
    sel = (ep == slate_id).nonzero(as_tuple=True)[0]
    return sel, cache["dv"][goal]


def rk_curve(dv_pred: torch.Tensor, dv_true: torch.Tensor, ks: list[int]):
    """Per-slate M_k, P_k for every K in ``ks``, RAW Lyapunov units.

        w_r(K)  = C(n-r, K-1) / C(n, K)                     model rank r's weight
        M_k     = sum_r t_[r]  w_r(K)   t_[r]  = true dV sorted ascending
                                                  (= oracle's expected pick
                                                  from a random K-subset)
        P_k     = sum_r t_(r)  w_r(K)   t_(r)  = true dV at the MODEL's rank r
                                                  (= model's expected pick)

    dV is a cost (lower is better), so M_k <= P_k always and R_k = M_k - P_k
    <= 0; |R_k| is the value left on the table by the model at pool size K.
    """
    n = dv_pred.shape[0]
    order = torch.argsort(dv_pred)
    t_by_rank = dv_true[order].numpy()
    t_sorted = torch.sort(dv_true).values.numpy()
    Mk, Pk = {}, {}
    for K in ks:
        if K > n:
            continue
        wr = _w(n, K, None)
        Mk[K] = float((t_sorted * wr).sum())
        Pk[K] = float((t_by_rank * wr).sum())
    return Mk, Pk


def _eval_cfg_path(cache):
    cfg = cache["config"]
    return cfg.get("eval_cfg") or cfg.get("slate_cfg")


@functools.lru_cache(maxsize=8)
def _wrapper_and_file_index(cfg_path: str, split: str = "train"):
    """Build the raw dataset once per (config, split) and index every row by
    its source file name, so ``load_occ0_for_slate`` below is a dict lookup
    on repeat calls against the same cache instead of a dataset rebuild."""
    from registry.dataset_registry import build_dataset
    from Genesis.training.dataset import PileSweepData

    cfg = yaml.safe_load(open(cfg_path).read())
    wrapper = build_dataset(cfg, split)
    raw = wrapper.raw_dataset

    files = []
    for path in cfg["paths"]:
        full = REPO_ROOT / "Genesis" / "data" / path
        runs = PileSweepData._collect_run_paths(PileSweepData, full)
        runs = PileSweepData._filter_split(runs, split, cfg.get("val_pct", 0),
                                            cfg.get("test_pct", 0))
        files.extend(str(d.name) for d, _ in runs)

    n = len(wrapper)
    file_to_rows: dict[str, list[int]] = {}
    for i in range(n):
        r = raw.get_run_index(i)
        file_to_rows.setdefault(files[r], []).append(i)
    return wrapper, raw, file_to_rows


def load_occ0_for_slate(cache, slate_id) -> torch.Tensor:
    """The step-0 occupancy every candidate of ``slate_id`` shares.

    A same-state slate's candidates all start from one pile, so any raw row
    whose source file matches one of this slate's cache rows gives the same
    occ0 -- which candidate within the file is arbitrary.
    """
    cfg_path = _eval_cfg_path(cache)
    wrapper, raw, file_to_rows = _wrapper_and_file_index(cfg_path)
    ep = cache["ep"]
    sel = (ep == slate_id).nonzero(as_tuple=True)[0]
    fname = cache["slate_files"][int(sel[0])]
    rows = file_to_rows.get(fname)
    if not rows:
        raise KeyError(f"no raw row found for source file {fname!r} "
                        f"(slate {slate_id}); rebuilt file index has "
                        f"{len(file_to_rows)} files, e.g. "
                        f"{list(file_to_rows)[:3]}")
    (input_grid, _physics), _target = raw[rows[0]]
    return input_grid[0]


def workspace_bounds(cache):
    cfg_path = _eval_cfg_path(cache)
    _wrapper, raw, _idx = _wrapper_and_file_index(cfg_path)
    return raw.workspace_bounds


def action_geometry(actions_row, ws_min, ws_max, grid_res):
    """One action [sx,sy,ex,ey] (metres) -> (start_px(col,row), end_px(col,row))
    using the identical convention as fit_linear_foresight.actions_to_pixels."""
    from fit_linear_foresight import actions_to_pixels
    a = actions_row.unsqueeze(0)
    s, e = actions_to_pixels(a, ws_min, ws_max, grid_res)
    return s[0], e[0]


def swept_rectangle_corners(start_px, end_px, half_width_px):
    """Four corners (col,row) of the swept rectangle for one push, so a
    caller can draw it with matplotlib.patches.Polygon."""
    s = np.asarray([float(start_px[0]), float(start_px[1])])
    e = np.asarray([float(end_px[0]), float(end_px[1])])
    d = e - s
    L = np.hypot(*d)
    if L < 1e-9:
        perp = np.array([0.0, 0.0])
    else:
        perp = np.array([-d[1], d[0]]) / L * half_width_px
    return np.array([s + perp, e + perp, e - perp, s - perp])


MODEL_COLORS = {
    "oracle": "#2ca02c",
    "UNet": "#1f77b4",
    "linear": "#d62728",
    "mean-delta": "#9467bd",
    "persistence": "#7f7f7f",
}
