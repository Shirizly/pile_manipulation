"""Chain-grouped, leakage-safe train/validation split of DS-0008 (EXP-0059
hyperparameter-tuning follow-up).

Every retrieval config knob (window, cap, corridor weight, k, aggregation)
must be tuned WITHOUT looking at DS-0009 (the register's held-out test set).
The only other chain-shaped corpus available is DS-0008 itself, so this
module holds out a fraction of its CHAINS (never individual rows -- a chain's
8 steps are a near-duplicate trajectory, so a row-random split would leak
almost the same transition into both the tuning bank and the validation
query set) and returns:

  * a TUNING BANK built from the REMAINING DS-0008 chains + all of DS-0010
    (DS-0010 rows are independent extracted transitions, not chains, so
    there is nothing to hold out there -- see its DATASET.md), in the exact
    shape `model.retrieval.bank.TransitionBank.build` produces;
  * a VALIDATION query set: the held-out DS-0008 chains, packaged as a list
    of per-file dicts in the SAME shape as `eval_narrow.py`'s `ch` (chain)
    list (`states, states_, p_starts, p_stops, chain_env, chain_step,
    start_kind, valid`), so `experiments/EXP-0059-*/code/eval_retrieval.py
    ::evaluate` can score `accuracy_1` / `rollout_accuracy_{1..4}` on it
    completely unchanged.

DS-0008 has no same-state ACTION POOLS (it is chain-shaped: one action per
step, not many candidate actions from one shared start), so `slateN` cannot
be validated this way -- only DS-0009's `test_pools` has that shape. The
sweep script therefore selects configs on validation `accuracy_1`/
`rollout_accuracy_4` only, and reports `slateN` alongside them purely as a
DS-0009 TEST number for the shortlisted best configs (never as a selection
criterion) -- this is a known, explicitly-scoped limitation, not an
oversight.
"""
from __future__ import annotations

import glob
import random
from typing import List, Tuple

import torch

from .bank import TransitionBank, DS0008_DIR, DS0010_DIR, _load_dir, DEFAULT_MOVED_THRESHOLD


def _subset_dict(d: dict, mask: torch.Tensor) -> dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.shape[:1] == mask.shape:
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.shape[0]:
            out[k] = [x for x, m in zip(v, mask.tolist()) if m]
        else:
            out[k] = v
    return out


def chain_holdout_split(val_frac: float = 0.2, seed: int = 0
                        ) -> Tuple[List[dict], List[dict]]:
    """-> (train_files, val_files): per-file dicts, DS-0008 rows only,
    split by GLOBAL chain id (`file_index * 1000 + chain_env`, safe since
    every file has exactly 32 chains). `valid=False` rows are dropped from
    both sides (never used for tuning OR validation, matching EXP-0053)."""
    files = sorted(glob.glob(str(DS0008_DIR / "_*_data.pt")))
    raw = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    all_ids = sorted({int(fi * 1000 + c) for fi, d in enumerate(raw)
                      for c in torch.unique(d["chain_env"]).tolist()})
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    n_val = max(1, int(round(len(all_ids) * val_frac)))
    val_ids = set(all_ids[:n_val])

    train_files, val_files = [], []
    for fi, d in enumerate(raw):
        v = d["valid"] if "valid" in d else torch.ones(len(d["states"]), dtype=torch.bool)
        gcid = fi * 1000 + d["chain_env"]
        is_val = torch.tensor([int(c) in val_ids for c in gcid.tolist()])
        train_mask = v & (~is_val)
        val_mask = v & is_val
        if bool(train_mask.any()):
            t = _subset_dict(d, train_mask)
            t["_file_index"] = fi
            train_files.append(t)
        if bool(val_mask.any()):
            sub = _subset_dict(d, val_mask)
            sub["_file_index"] = fi
            # `eval_retrieval.py::evaluate`'s rollout loop does
            # `for e in range(chain_env.max()+1): rows = nonzero(chain_env==e)`
            # and indexes `rows[0]` unconditionally -- it assumes every e in
            # that range has at least one row, true for a full corpus file
            # but NOT for an arbitrary held-out chain subset (which leaves
            # gaps). Remap this file's surviving chain_env values to a
            # contiguous 0..k-1 range so the assumption holds again.
            sub["_orig_chain_env"] = sub["chain_env"].clone()   # kept for tests only
            uniq = torch.unique(sub["chain_env"])
            remap = {int(c): i for i, c in enumerate(uniq.tolist())}
            sub["chain_env"] = torch.tensor([remap[int(c)] for c in sub["chain_env"].tolist()])
            val_files.append(sub)
    return train_files, val_files


def build_tuning_bank_and_val_chains(val_frac: float = 0.2, seed: int = 0,
                                     moved_threshold: float = DEFAULT_MOVED_THRESHOLD,
                                     include_ds0010: bool = True
                                     ) -> Tuple[TransitionBank, List[dict]]:
    """The one call sites need: a `TransitionBank` built from DS-0008's
    TRAIN-side chains (+ all of DS-0010) plus the held-out DS-0008 chains as
    a validation query set, in `eval_retrieval.py`'s chain-dict shape."""
    train_files, val_files = chain_holdout_split(val_frac, seed)
    s0 = torch.cat([d["states"] for d in train_files]).float()
    s1 = torch.cat([d["states_"] for d in train_files]).float()
    ps = torch.cat([d["p_starts"] for d in train_files]).float()
    pe = torch.cat([d["p_stops"] for d in train_files]).float()
    source = ["DS-0008-train"] * len(s0)
    if include_ds0010:
        s0b, s1b, psb, peb = _load_dir(str(DS0010_DIR / "*_data.pt"), has_valid=False)
        s0 = torch.cat([s0, s0b]); s1 = torch.cat([s1, s1b])
        ps = torch.cat([ps, psb]); pe = torch.cat([pe, peb])
        source += ["DS-0010"] * len(s0b)
    bank = TransitionBank.from_states(s0, s1, ps, pe, source=source, moved_threshold=moved_threshold)
    return bank, val_files
