"""Baselines/common/data.py -- one loader, two views.

Every baseline model needs the SAME transitions in two representations:

  * a **particle view** -- per-transition ``states``/``states_`` ((20,7):
    xyz + quat, world metres), ``p_start``/``p_stop`` ((3,), world metres),
    ``angle`` (scalar), for particle-space models (e.g. the GNN);
  * an **occupancy view** -- ``occ0``/``occ1`` at 64x64, for grid models
    (UNet/CNN/etc).

The occupancy view MUST be byte-identical to every number already recorded
in docs/experiments/REGISTER.md, so it is produced by going through
``registry.dataset_registry.build_dataset`` on the SAME dataset configs
``scripts/probes/expB_multistep_eval.py`` uses -- not a new rasteriser.

The particle view is read from the SAME underlying run arrays
(``states``/``states_``/``p_starts``/``p_stops``/``angles``, loaded once per
source ``_<batch>_data.pt`` file by ``Genesis.training.dataset.PileSweepData``
itself) using PileSweepData's OWN index bookkeeping
(``get_run_index`` + the two-line ``_resolve_idx``/``_run_lookup``/
``_offsets`` indirection that ``PileSweepData.get_raw_action`` and
``_extract_sample_in_pxl`` already use internally -- see
``Genesis/training/dataset.py`` lines ~150-169). This is NOT a re-derivation
of frame conventions (there is a documented history of exactly that kind of
bug -- C-018, the axis-transpose fix described in
``Genesis/training/dataset.py``'s own ``_draw_particle_grid`` docstring): it
is the two lines of index arithmetic the library already performs, applied
here so a particle-view row and an occupancy-view row at the same index
`i` are guaranteed to be the same transition.

Do NOT invent a new rasteriser either. ``rasterize_particles`` below calls
``PileSweepData._draw_particle_grid`` directly -- private (there is no public
wrapper), but the EXACT OpenCV routine that produced every ground-truth
occ1 in this dataset, axis-convention bug already fixed there. A candidate
public alternative, ``transforms.functional.particles_to_occupancy``, is a
point-splat rasteriser with a DIFFERENT convention (no box/quaternion
handling) used by other parts of the repo (e.g. the MPC oracle) -- it is
NOT a drop-in replacement and using it here would reintroduce the class of
bug ``_draw_particle_grid``'s docstring documents. See ``LOG.md``.

Manifest note: ``manifest.json`` lives in the SOURCE cell (e.g.
``Genesis/data/slates_multistep/n20_L20mm/manifest.json``), not in the
``_train``/``_eval`` symlink folders, and maps
``batch_idx -> {slate_idx, step_idx}``. Slate/step tagging therefore only
makes sense for a single-source-cell config (an ``_eval`` cell, or a
``_train`` cell); the pooled ``n20_L20L40`` train config spans two source
cells with two separate manifests and is never tagged (nor does it need to
be -- it is only ever used to FIT an operator or TRAIN a model, both of
which are manifest-agnostic).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from registry.dataset_registry import build_dataset
from scripts.probes.expB_multistep_eval import _run_files


@dataclass
class CellData:
    tag: str
    occ0: torch.Tensor        # (N,H,W) float32 in [0,1]
    occ1: torch.Tensor        # (N,H,W) float32 in [0,1] -- GROUND TRUTH, eval-only
    actions: torch.Tensor     # (N,4) world [sx,sy,ex,ey] metres
    states: torch.Tensor      # (N,20,7) pre-push particle xyz+quat, world metres
    states_: torch.Tensor     # (N,20,7) post-push particle xyz+quat -- GROUND TRUTH, eval-only
    p_start: torch.Tensor     # (N,3) world metres
    p_stop: torch.Tensor      # (N,3) world metres
    angle: torch.Tensor       # (N,)
    run_idx: torch.Tensor     # (N,) long -- index into `files`/`raw.configs` (this split's sorted file order)
    slate_idx: torch.Tensor   # (N,) long, -1 if manifest_path was not given
    step_idx: torch.Tensor    # (N,) long, -1 if manifest_path was not given
    files: list                # per-run filename, files[run_idx[i]] is row i's source file
    workspace_min: tuple
    workspace_max: tuple
    H: int
    W: int
    raw: object                # the PileSweepData instance (kept for rasterize_particles)


def _resolve_sample(raw, i: int):
    """(run_idx, sample_index_within_run) for row `i`.

    Literally the two lines ``PileSweepData.get_raw_action`` /
    ``_extract_sample_in_pxl`` already use internally
    (``Genesis/training/dataset.py``) to go from a flat dataset index to
    (run, sample-within-run) -- repeated here (not imported: they are
    inlined in those methods, not factored out) so the particle view can
    pull ``states``/``states_``/``angles`` for row `i`, which those methods
    do not expose.
    """
    idx = raw._resolve_idx(i)
    run_idx = raw._run_lookup[idx]
    sample_index = idx - raw._offsets[run_idx]
    return run_idx, sample_index


def load_cell(cfg_path: str, split: str = "train", manifest_path: str | None = None,
              tag: str | None = None) -> CellData:
    """Build both views for one dataset config.

    `cfg_path` is any ``configs/dataset/genesis_slates_multistep_*.yaml``
    (a per-cell ``_train``/``_eval`` config, or the pooled
    ``n20_L20L40_train`` config). `split` matches the config's own
    val_pct/test_pct=0 convention ("train" loads every file, see any of
    those configs' header comments).

    `manifest_path` (the SOURCE cell's manifest.json) is required to get
    non-trivial `slate_idx`/`step_idx` -- pass it for an eval cell (needed
    downstream for the step-0-only control-utility filter); omit it (or
    pass None) when only fitting/training, where slate/step tags are unused.
    """
    cfg = yaml.safe_load(open(cfg_path).read())
    wrapper = build_dataset(cfg, split)
    raw = wrapper.raw_dataset
    n = len(wrapper)

    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    run_idx = torch.tensor([raw.get_run_index(i) for i in range(n)], dtype=torch.long)

    states = torch.empty((n, 20, 7), dtype=torch.float32)
    states_ = torch.empty((n, 20, 7), dtype=torch.float32)
    p_start = torch.empty((n, 3), dtype=torch.float32)
    p_stop = torch.empty((n, 3), dtype=torch.float32)
    angle = torch.empty((n,), dtype=torch.float32)
    for i in range(n):
        r, s = _resolve_sample(raw, i)
        run = raw.runs[r]
        states[i] = run["states"][s]
        states_[i] = run["states_"][s]
        p_start[i] = run["p_starts"][s]
        p_stop[i] = run["p_stops"][s]
        angle[i] = run["angles"][s]

    files = _run_files(cfg_path, split)
    slate_idx = torch.full((n,), -1, dtype=torch.long)
    step_idx = torch.full((n,), -1, dtype=torch.long)
    if manifest_path is not None:
        manifest = json.loads(open(manifest_path).read())
        batch_lookup = {b["batch_idx"]: (b["slate_idx"], b["step_idx"]) for b in manifest["batches"]}
        run_to_batch = {r: int(fn.split("_")[1]) for r, fn in enumerate(files)}
        slate_idx = torch.tensor([batch_lookup[run_to_batch[int(r)]][0] for r in run_idx.tolist()],
                                  dtype=torch.long)
        step_idx = torch.tensor([batch_lookup[run_to_batch[int(r)]][1] for r in run_idx.tolist()],
                                 dtype=torch.long)

    ws_min, ws_max = raw.workspace_bounds
    H, W = occ0.shape[-2:]
    return CellData(tag=tag or Path(cfg_path).stem, occ0=occ0, occ1=occ1, actions=actions,
                     states=states, states_=states_, p_start=p_start, p_stop=p_stop,
                     angle=angle, run_idx=run_idx, slate_idx=slate_idx, step_idx=step_idx,
                     files=files, workspace_min=ws_min, workspace_max=ws_max, H=H, W=W, raw=raw)


def rasterize_particles(raw, run_idx: int, particles_world: torch.Tensor) -> torch.Tensor:
    """Rasterise ONE transition's (20,7) world-frame particle states
    [xyz + quat] to a (H,W) occupancy grid, using PileSweepData's own
    ``_draw_particle_grid`` for the config of source run `run_idx` -- the
    exact routine that produced every ground-truth occ0/occ1 in this
    dataset. `particles_world` must be in the SAME world-metre frame as
    `CellData.states`/`states_` (the routine itself converts to pixels via
    `raw.to_pxl`/`raw.ctr_in_PXL`, exactly as `PileSweepData.__getitem__` does).

    A particle-space predictor calls this once per row in its batch (the
    routine is single-transition; there is no batched form in the
    original either).
    """
    particles_px = particles_world.clone()
    particles_px[:, :3] = particles_px[:, :3] * raw.to_pxl + raw.ctr_in_PXL
    grid = torch.zeros((raw._output_grid.shape[0], raw._output_grid.shape[1]), dtype=torch.float32)
    raw._draw_particle_grid(particles_px, grid, raw.configs[run_idx])
    return grid
