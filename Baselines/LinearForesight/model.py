"""Baselines/LinearForesight/model.py -- apply a SWITCHED (per-length-bin)
set of pixel-space linear operators, choosing the operator per-row from the
row's own push length.

This is the inference half of Suh & Tedrake 2020's switched-linear model
(`docs/linear_visual_foresight_baseline.md` Step 2a): the single global
operator `Baselines/common/eval_baseline.py`/`eval_randlen_indist.py` already
fit and used as the "linear" reference row is ONE `A` shared by every push
length. The paper's own design switches `A` by a discretized push-length bin
(their 5 bins); this module is the length-only switching rule (their `l`
axis) applied to whatever bin edges/operators `fit_switched.py` produced.

No new warp/apply/unwarp/blend math here -- that primitive already exists
(`transforms/functional.py`'s `to_push_frame`/`from_push_frame`/
`push_frame_validity_mask`/`blend_push_prediction`, wired up as
`fit_linear_foresight.predict_world`). This module only adds the bin
selection and the per-bin dispatch, reusing `predict_world` per-bin exactly
as fitting reuses `fit_operator`/`fit_operator_nonneg` per-bin.
"""
from __future__ import annotations

import torch

from fit_linear_foresight import predict_world


def push_length_m(actions: torch.Tensor) -> torch.Tensor:
    """World-metre push length from `[sx,sy,ex,ey]` actions -- resolution-
    independent, unlike a pixel-space length, so it is the same quantity at
    fit time and at predict time regardless of `--res`/`resolution_scale`."""
    return (actions[:, 2:4] - actions[:, 0:2]).norm(dim=-1)


def bin_index(lengths_m: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Bucketize `lengths_m` into `len(bin_edges) - 1` bins using the
    INTERIOR edges only (`bin_edges[1:-1]`), so a length below `bin_edges[0]`
    or above `bin_edges[-1]` still lands in the first/last bin rather than
    raising or falling outside `[0, n_bins)` -- exactly `torch.bucketize`'s
    clamping behaviour, made explicit here since it is easy to get backwards
    (using the full `bin_edges` would silently return one bin too many)."""
    return torch.bucketize(lengths_m, bin_edges[1:-1].to(lengths_m.device))


def predict_switched(bin_edges: torch.Tensor, operators: list[torch.Tensor],
                      occ: torch.Tensor, start_px: torch.Tensor, end_px: torch.Tensor,
                      lengths_m: torch.Tensor, res: int, grid_res: tuple[int, int],
                      scale: float = 1.0) -> torch.Tensor:
    """Predict a whole batch, dispatching each row to its own bin's operator.

    Rows are grouped by bin (not looped one at a time) so each bin still
    pays `predict_world`'s batched warp cost, not a per-row one. A bin with
    zero rows in this call is simply never touched -- there is nothing to
    predict for it.
    """
    out = occ.clone()
    bins = bin_index(lengths_m, bin_edges)
    for b, A in enumerate(operators):
        m = bins == b
        if not bool(m.any()):
            continue
        out[m] = predict_world(A.to(occ.device), occ[m], start_px[m], end_px[m],
                                res, grid_res, scale)
    return out
