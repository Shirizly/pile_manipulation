"""Shared descriptor_accuracy metric for the dmdc-lenbins basis-comparison
experiment (variants B and C use the SAME definition):

    descriptor_accuracy = 1 - rms(model_err) / rms(persistence_err)

where every descriptor dim is z-scored by the TRAIN-set mean/std of phi_t1
before the rms, the `const` block is excluded from the AGGREGATE, and
per-block accuracies are reported under the same normalisation.

Written as metric_c.py (not metric.py) because variant B's agent owns
metric.py; this file exists in case metric.py is not present when variant C
runs. If metric.py exists, prefer importing it directly instead.
"""
from __future__ import annotations

import torch


def zscore_stats(phi_t1_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std of phi_t1 computed on the TRAIN split only."""
    mu = phi_t1_train.mean(dim=0)
    sigma = phi_t1_train.std(dim=0).clamp_min(1e-8)
    return mu, sigma


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x ** 2).mean()))


def descriptor_accuracy(pred: torch.Tensor, truth: torch.Tensor, persist: torch.Tensor,
                         mu: torch.Tensor, sigma: torch.Tensor,
                         const_slice: slice | None = None) -> float:
    """Aggregate accuracy, z-scored by (mu, sigma), const block excluded."""
    predz = (pred - mu) / sigma
    truthz = (truth - mu) / sigma
    persz = (persist - mu) / sigma
    if const_slice is not None:
        D = pred.shape[1]
        keep = torch.ones(D, dtype=torch.bool)
        keep[const_slice] = False
        predz, truthz, persz = predz[:, keep], truthz[:, keep], persz[:, keep]
    model_err = predz - truthz
    persist_err = persz - truthz
    return 1.0 - rms(model_err) / max(rms(persist_err), 1e-12)


def block_accuracy(pred: torch.Tensor, truth: torch.Tensor, persist: torch.Tensor,
                    mu: torch.Tensor, sigma: torch.Tensor,
                    slices: dict[str, slice]) -> dict[str, float]:
    """Per-block accuracy under the SAME z-scoring (const included here, since
    only the AGGREGATE excludes it; const will show as degenerate/blown-up,
    same caveat variant A flagged -- persistence error there is ~0 by
    construction because the const dim is literally always 1)."""
    out = {}
    for name, sl in slices.items():
        if name == "_total":
            continue
        predz = (pred[:, sl] - mu[sl]) / sigma[sl]
        truthz = (truth[:, sl] - mu[sl]) / sigma[sl]
        persz = (persist[:, sl] - mu[sl]) / sigma[sl]
        model_err = predz - truthz
        persist_err = persz - truthz
        out[name] = 1.0 - rms(model_err) / max(rms(persist_err), 1e-12)
    return out
