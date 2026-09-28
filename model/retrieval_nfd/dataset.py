"""model/retrieval_nfd/dataset.py -- flat, precomputed-cache Dataset for the
"NFD with a retrieved reference" model (EXP-0059 section 8).

Reads `model/retrieval_nfd/cache/retrieval_ref_cache.pt` (built once by
`precompute.py`) and returns EXACTLY the `((input_grid, physics), target)`
tuple contract `Genesis.training.dataset.PileSweepData.__getitem__` returns,
so it plugs into `registry.dataset_registry.EulerianDatasetWrapper` the same
way `Baselines.NFD.nfd_lib.PileSweepData3Ch` does -- no Trainer/registry code
needs to know this dataset does not touch disk-per-sample or Genesis at all.

5 input channels: [occ0, r_start, r_stop, donor_occ0, donor_occ1] (the
design doc's mandatory set; the optional 6th "retrieval clean prediction"
channel is NOT implemented here -- out of scope for the coder's time
budget, noted in `coder_status.md`).

Donor selection: `donor_field="main"` picks uniformly at random, EACH
`__getitem__` CALL (i.e. effectively each epoch, since the DataLoader calls
this once per row per epoch), among the row's cached top-3 donor candidates
(`donor_idx3`, already excluding the row's own chain -- see `donors.py`).
`donor_field="random"` always uses the single cached bucket-matched random
donor (the control model). Reference dropout (`dropout_p`, default 0.15)
zeroes channels 3/4 with that probability, independent of which donor field
is selected -- "test-time zeroed reference" is a separate, eval-side control
(see `predictor.py`), not this flag.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "retrieval_ref_cache.pt"


class RetrievalRefFlatDataset(Dataset):
    def __init__(self, split: str, cache_path: str | None = None,
                 donor_field: str = "main", dropout_p: float = 0.15,
                 seed: int = 0):
        assert split in ("train", "val", "test")
        assert donor_field in ("main", "random")
        path = Path(cache_path) if cache_path else CACHE_PATH
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        splits = d["split"]
        idx = [i for i, s in enumerate(splits) if s == split]
        assert idx, f"no rows in split={split!r} (cache has {len(splits)} rows total)"
        self.idx = idx
        self.occ0 = d["occ0"][idx].float()
        self.occ1 = d["occ1"][idx].float()          # target
        self.r_start = d["r_start"][idx].float()
        self.r_stop = d["r_stop"][idx].float()
        self.donor_occ0_3 = d["donor_occ0"][idx].float()   # (n, 3, H, W)
        self.donor_occ1_3 = d["donor_occ1"][idx].float()
        self.rand_occ0 = d["rand_occ0"][idx].float()
        self.rand_occ1 = d["rand_occ1"][idx].float()
        self.donor_field = donor_field
        self.dropout_p = float(dropout_p)
        self.split = split
        self._g = torch.Generator().manual_seed(seed + (0 if split == "train" else 12345))
        self.H = self.occ0.shape[-2]
        self.W = self.occ0.shape[-1]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        occ0 = self.occ0[i]
        r_start = self.r_start[i]
        r_stop = self.r_stop[i]
        target = self.occ1[i]
        if self.donor_field == "main":
            j = int(torch.randint(3, (1,), generator=self._g))
            donor0 = self.donor_occ0_3[i, j]
            donor1 = self.donor_occ1_3[i, j]
        else:
            donor0 = self.rand_occ0[i]
            donor1 = self.rand_occ1[i]
        if self.dropout_p > 0 and float(torch.rand(1, generator=self._g)) < self.dropout_p:
            donor0 = torch.zeros_like(donor0)
            donor1 = torch.zeros_like(donor1)
        input_grid = torch.stack([occ0, r_start, r_stop, donor0, donor1], dim=0)  # (5, H, W)
        physics = torch.zeros(1, dtype=torch.float32)  # unused; uses_physics=False
        return (input_grid, physics), target
