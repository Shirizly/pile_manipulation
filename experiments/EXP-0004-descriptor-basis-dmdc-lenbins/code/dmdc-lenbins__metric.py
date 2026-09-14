"""Shared descriptor_accuracy metric for the dmdc-lenbins temp experiment
(variants A/B/C all score through this module so their numbers are comparable).

DEFINITION (read this before touching any results file):

  1. Z-SCORE every descriptor dimension by the TRAIN-set per-dimension mean
     and std of `phi_t1` (the one-step-ahead target), computed ONCE on train
     and reused (frozen) to normalise train, test, predictions and the
     persistence baseline alike. Std is floored at 1e-8 to avoid blow-ups on
     near-constant dims.
  2. In that normalised space:
         descriptor_accuracy = 1 - rms(model_err) / rms(persistence_err)
     where model_err = z(pred) - z(truth), persistence_err = z(phi_t) - z(truth),
     rms is over ALL selected dims and ALL rows pooled (not averaged-of-per-dim
     rms -- one pooled rms top and bottom, matching variant A's original
     convention, just now on normalised units).
  3. The `const` descriptor block (dmdc_baseline's block name "const", a
     literal 1.0 column) is DROPPED from every accuracy computed here --
     overall and per-block alike. It is degenerate: persistence error on a
     constant is ~0, so it either contributes nothing or (after z-scoring,
     where std->1e-8 floor kicks in) blows up the ratio. `exclude_blocks`
     defaults to {"const"} for exactly this reason -- don't pass an empty set
     unless you have a specific reason to include it.
  4. Per-block accuracy uses the SAME z-scored space and the SAME formula,
     just restricted to that block's dims (still pooled rms within the
     block, across rows).

Why z-score by phi_t1 train stats specifically: descriptor dims have wildly
different natural scales and near-persistence-perfect dims (e.g. mass, which
is ~conserved) would otherwise dominate the denominator and mask real signal
or noise in other blocks (this is exactly the problem the un-normalised
variant-A headline had -- see RESULTS.md "Fix the aggregate metric").
Z-scoring by phi_t1 (not phi_t) is a choice: it's the target distribution,
so every dim is judged relative to how much it actually varies target-side.
"""
from __future__ import annotations

import torch


def fit_zscore(phi_t1_train: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension (mu, sigma) from the TRAIN split's phi_t1. Freeze and reuse."""
    mu = phi_t1_train.mean(dim=0)
    sigma = phi_t1_train.std(dim=0).clamp_min(eps)
    return mu, sigma


def zscore(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sigma


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x ** 2).mean()))


def descriptor_accuracy(
    pred: torch.Tensor, truth: torch.Tensor, persist: torch.Tensor,
    mu: torch.Tensor, sigma: torch.Tensor,
    slices: dict[str, slice] | None = None,
    exclude_blocks: set[str] = frozenset({"const"}),
) -> dict:
    """Compute overall + per-block descriptor_accuracy in z-scored space.

    pred/truth/persist: [N, D] raw (un-normalised) descriptor arrays, same D
    as mu/sigma. `slices` (e.g. dmdc_baseline.descriptor_slices(n_fourier),
    or a merged variant-A+B layout) drives the per-block breakdown; blocks in
    `exclude_blocks` are skipped for both the breakdown AND removed from the
    dims used in the "overall" number. Pass `slices=None` to skip the
    per-block breakdown (overall only, over ALL dims -- caller must have
    already excluded any degenerate dims from pred/truth/persist/mu/sigma).

    Returns {"overall": float, "blocks": {name: float}}.
    """
    zp = zscore(pred, mu, sigma)
    zt = zscore(truth, mu, sigma)
    zs = zscore(persist, mu, sigma)

    if slices is None:
        model_err = zp - zt
        persist_err = zs - zt
        overall = 1.0 - _rms(model_err) / max(_rms(persist_err), 1e-12)
        return {"overall": overall, "blocks": {}}

    keep_dims = []
    blocks: dict[str, float] = {}
    for name, sl in slices.items():
        if name.startswith("_"):
            continue
        m_err = zp[:, sl] - zt[:, sl]
        p_err = zs[:, sl] - zt[:, sl]
        blocks[name] = 1.0 - _rms(m_err) / max(_rms(p_err), 1e-12)
        if name not in exclude_blocks:
            keep_dims.extend(range(sl.start, sl.stop))

    idx = torch.tensor(keep_dims, dtype=torch.long)
    model_err = zp[:, idx] - zt[:, idx]
    persist_err = zs[:, idx] - zt[:, idx]
    overall = 1.0 - _rms(model_err) / max(_rms(persist_err), 1e-12)
    return {"overall": overall, "blocks": blocks}
