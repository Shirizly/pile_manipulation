"""Baselines/common/randlen_data.py -- N-agnostic loader for the
overnight_randlen corpus (mixed_n20, piled_n20, scattered_n20, piled_n50,
scattered_n50).

Builds the same occupancy/action view `Baselines.common.data.load_cell`/
`CellData` does, EXCEPT `states`/`states_` -- those have a per-cube shape
that differs between the n20 and n50 groups, and `CellData`'s (n,20,7)
preallocation crashes the moment it hits an n50 file. Nothing in the
cross-corpus accuracy/capture report needs `states`/`states_`: image
accuracy compares occ0/occ1 directly, and GNN's own node positions come
from occ0 via `Baselines/GNN/perception.py` (privileged simulator state is
used only at TRAINING time, to build a label -- never at eval/inference,
see `Baselines/GNN/SPEC.md`'s "CORRECTION" section). So this loader can
skip the field that breaks on heterogeneous true-N entirely, rather than
needing to fix `CellData` itself.

The returned object exposes the same field names `Baselines.common.
eval_baseline.PredictorBatch` does (`occ0`, `p_start`, `p_stop`, `angle`,
`run_idx`, `raw`, ...), so both `Baselines/GNN/predictor.py` and
`Baselines/NFD/predictor.py` can be called with it directly -- no adapter
needed.

Slate/step structure: every file in this corpus is a same-state pool of
128 step-0 candidates (`Genesis/data/overnight_randlen/DATASET.yaml`,
verified spread 0/1.16e-10 m across envs), env-major blocked (row r within
a file = env r%128, step r//128) -- exactly the convention
`Baselines/common/eval_randlen_indist.py` already derives by hand (no
manifest.json exists for this corpus), reused here rather than re-derived
differently.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import yaml

from Baselines.common.data import _resolve_sample
from registry.dataset_registry import build_dataset

POOL_SIZE = 128  # envs per same-state pool, this corpus's own collection convention


@dataclass
class RandlenCellData:
    tag: str
    occ0: torch.Tensor        # (N,H,W)
    occ1: torch.Tensor        # (N,H,W) -- ground truth, eval-only
    actions: torch.Tensor     # (N,4) world [sx,sy,ex,ey] metres
    p_start: torch.Tensor     # (N,3) world metres
    p_stop: torch.Tensor      # (N,3) world metres
    angle: torch.Tensor       # (N,)
    run_idx: torch.Tensor     # (N,) long -- index into raw.configs/this split's file order
    slate_idx: torch.Tensor   # (N,) long -- = run_idx (one file == one same-state pool)
    step_idx: torch.Tensor    # (N,) long -- (row within run) // 128
    files: list
    workspace_min: tuple
    workspace_max: tuple
    H: int
    W: int
    raw: object                # the PileSweepData instance


def load_randlen_cell(cfg_path: str, split: str = "train", tag: str | None = None,
                       need_step_idx: bool = True) -> RandlenCellData:
    """`need_step_idx=False` skips the `p_start`/`p_stop`/`angle`/`step_idx`
    per-row loop and its completeness assertion entirely (fields come back
    as size-0 placeholders, the same -1/empty convention `Baselines.common.
    data.CellData` uses when its own `manifest_path` is omitted). Use this
    for callers that only need `occ0`/`occ1`/`actions` (e.g. fitting a
    transition operator over the whole corpus) -- the assertion below is a
    real correctness check for the step-0-candidate-ranking use case
    (`Baselines/common/eval_report.py`, `eval_randlen_indist.py`'s own
    inline derivation), but it can legitimately fail on the pooled
    `_all.yaml` configs, where a `min_push_length_m` filter dropping even
    one row anywhere in a 512-row file breaks the `sample_idx // POOL_SIZE`
    step heuristic for the rest of that file -- a real data fact, not a bug
    to silence, but not every caller needs step tagging to be correct."""
    cfg = yaml.safe_load(open(cfg_path).read())
    wrapper = build_dataset(cfg, split)
    raw = wrapper.raw_dataset
    n = len(wrapper)

    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    run_idx = torch.tensor([raw.get_run_index(i) for i in range(n)], dtype=torch.long)

    if need_step_idx:
        p_start = torch.empty((n, 3), dtype=torch.float32)
        p_stop = torch.empty((n, 3), dtype=torch.float32)
        angle = torch.empty((n,), dtype=torch.float32)
        step_idx = torch.empty((n,), dtype=torch.long)
        for i in range(n):
            r, s = _resolve_sample(raw, i)
            run = raw.runs[r]
            p_start[i] = run["p_starts"][s]
            p_stop[i] = run["p_stops"][s]
            angle[i] = run["angles"][s]
            step_idx[i] = s // POOL_SIZE

        slate_idx = run_idx.clone()

        # Sanity check, matching eval_randlen_indist.py's own guard: a
        # min_push_length_m filter dropping a step-0 row would silently break
        # the `s // POOL_SIZE` derivation above.
        step0 = step_idx == 0
        n_files = int(slate_idx.unique().numel())
        n_expect = n_files * POOL_SIZE
        n_s0 = int(step0.sum())
        assert n_s0 == n_expect, (
            f"{cfg_path}: step-0 subset is {n_s0}, expected exactly {n_expect} "
            f"({n_files} files x {POOL_SIZE}) -- a min_push_length_m filter "
            f"must have dropped a step-0 row; the step_idx = sample_idx // "
            f"{POOL_SIZE} derivation assumes it did not. Pass "
            f"need_step_idx=False if the caller does not need step tagging.")
    else:
        p_start = torch.empty((0, 3), dtype=torch.float32)
        p_stop = torch.empty((0, 3), dtype=torch.float32)
        angle = torch.empty((0,), dtype=torch.float32)
        step_idx = torch.full((n,), -1, dtype=torch.long)
        slate_idx = torch.full((n,), -1, dtype=torch.long)

    ws_min, ws_max = raw.workspace_bounds
    H, W = occ0.shape[-2:]
    return RandlenCellData(tag=tag or cfg_path, occ0=occ0, occ1=occ1, actions=actions,
                            p_start=p_start, p_stop=p_stop, angle=angle,
                            run_idx=run_idx, slate_idx=slate_idx, step_idx=step_idx,
                            files=None, workspace_min=ws_min, workspace_max=ws_max,
                            H=H, W=W, raw=raw)
