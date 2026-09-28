"""model/retrieval_nfd/lib.py -- registers the "NFD with a retrieved
reference" dataset type (EXP-0059 section 8).

Model side reuses `Baselines.NFD.nfd_lib`'s EXISTING generic `nfd-unet3ch`
model factory unchanged (it already forwards `in_channels` from the config
dict, see that module's docstring) -- config just sets `in_channels: 5`.
Only a new DATASET type is needed here, backed by `dataset.
RetrievalRefFlatDataset` (a precomputed flat cache, see `precompute.py`),
wrapped in the same `EulerianDatasetWrapper` every other NFD variant uses.

Import this module (registers `nfd-genesis-retrieval-ref`) before
`Trainer.from_config` looks the dataset type up -- see `train_retrieval_nfd.py`.
"""
from __future__ import annotations

from registry.dataset_registry import EulerianDatasetWrapper, register_dataset
from model.retrieval_nfd.dataset import RetrievalRefFlatDataset


@register_dataset("nfd-genesis-retrieval-ref")
def _build_retrieval_ref_dataset(cfg: dict, split: str) -> EulerianDatasetWrapper:
    """Config keys: cache_path (default: `dataset.CACHE_PATH`), donor_field
    ("main" | "random" -- "random" is the random-donor control model),
    dropout_p (default 0.15), transforms (same as `nfd_3ch_narrow_l20.yaml`:
    ensure_representation eulerian + eulerian_aliases). `val_pct`/`test_pct`
    are NOT read here -- the cache's own `split` tag (baked in by
    `precompute.py`, using `PileSweepData._filter_split` with val_pct=5/
    test_pct=5 over the SAME two directories `nfd_3ch_narrow_l20.yaml`
    trains on) is authoritative, so config values are ignored rather than
    silently re-splitting differently."""
    raw = RetrievalRefFlatDataset(
        split=split,
        cache_path=cfg.get("cache_path"),
        donor_field=cfg.get("donor_field", "main"),
        dropout_p=float(cfg.get("dropout_p", 0.15)),
        seed=int(cfg.get("seed", 0)),
    )
    return EulerianDatasetWrapper(
        raw, include_physics=bool(cfg.get("include_physics", False)),
        transforms_cfg=cfg.get("transforms"),
    )
