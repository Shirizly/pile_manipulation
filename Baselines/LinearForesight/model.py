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


# ---------------------------------------------------------------------------
# Differentiable (soft) length gate -- EXP-0023
# ---------------------------------------------------------------------------
#
# `bin_index` above is a `torch.bucketize` hard gate: correct for offline
# scoring, but useless as a SOURCE OF GRADIENTS. Under gradient-descent MPC the
# push length moves continuously; d(bin)/d(length) is zero almost everywhere
# and undefined at the edges, so the operator choice contributes nothing to
# dL/d(action) and the length gradient is systematically wrong near a boundary.
#
# `soft_bin_weights` replaces the argmax-over-bins by a triangular ("hat")
# interpolation between the two ADJACENT BIN CENTRES, which is exactly linear
# interpolation of the per-bin operators as a function of push length:
#
#     t      = (L - c_0) / w            continuous bin coordinate, c_b = centre
#     weight = relu(1 - |t - b|)        for bin b; nonzero for at most 2 bins
#
# At a bin centre the weights collapse to a one-hot on that bin, so the soft
# gate AGREES EXACTLY with the hard gate at every centre and differs in between
# (at a boundary it is the 50/50 average of the two neighbours rather than a
# jump). t is clamped to [0, n_bins-1], so lengths outside the fitted range
# fall back to the end bin, matching `bin_index`'s own clamping.
#
# THIS IS A RELAXATION OF THE FITTED MODEL, NOT THE FITTED MODEL. The hard gate
# is what every register row was measured with and is left untouched; anything
# scored through the soft gate must say so.

def bin_centres(bin_edges: torch.Tensor) -> torch.Tensor:
    """(n_bins,) centres of the `len(bin_edges)-1` bins."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def soft_bin_weights(lengths_m: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """(B,) push lengths -> (B, n_bins) differentiable, rows summing to 1.

    Triangular interpolation between adjacent bin centres (see module note).
    Assumes the equal-width bins `fit_switched.py` produces; the width is
    taken from the first bin.
    """
    edges = bin_edges.to(lengths_m.device, lengths_m.dtype)
    centres = bin_centres(edges)                       # (n_bins,)
    n_bins = centres.numel()
    width = (edges[1] - edges[0]).clamp_min(1e-12)
    t = ((lengths_m - centres[0]) / width).clamp(0.0, float(n_bins - 1))   # (B,)
    idx = torch.arange(n_bins, device=lengths_m.device, dtype=lengths_m.dtype)
    w = (1.0 - (t[:, None] - idx[None, :]).abs()).clamp_min(0.0)           # (B,n_bins)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


def predict_switched_soft(bin_edges: torch.Tensor, operators: list[torch.Tensor],
                          occ: torch.Tensor, start_px: torch.Tensor, end_px: torch.Tensor,
                          lengths_m: torch.Tensor, res: int, grid_res: tuple[int, int],
                          scale: float = 1.0) -> torch.Tensor:
    """Soft-gated twin of `predict_switched`, differentiable end-to-end.

    Differs from `predict_switched` in two ways beyond the gate:

    * the warp is paid ONCE for the whole batch and every operator is applied
      to the same canonical occupancy, then length-weighted -- so the cost is
      `n_bins` cheap matmuls, not `n_bins` warps;
    * nothing is written in place into a clone (`predict_switched`'s
      ``out[m] = ...``), because the result is a single expression the whole
      batch flows through -- no masked index assignment for autograd to have
      to survive.
    """
    from transforms.functional import push_frame_roundtrip

    w = soft_bin_weights(lengths_m, bin_edges)                     # (B,n_bins)
    ops = [A.to(device=occ.device, dtype=occ.dtype) for A in operators]

    def apply_soft(canon):
        flat = canon.reshape(canon.shape[0], -1)                   # (B,res*res)
        stack = torch.stack([(A @ flat.T).T[:, :res * res] for A in ops], dim=1)
        return (w[:, :, None] * stack).sum(dim=1).reshape(-1, res, res)

    pred = push_frame_roundtrip(apply_soft, occ, start_px, end_px, res,
                                scale=scale)
    return pred.clamp(0.0, 1.0)
