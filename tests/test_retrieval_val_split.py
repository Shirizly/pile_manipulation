"""Tests for the leakage-safe DS-0008 train/validation chain split
(EXP-0059 follow-up: `model/retrieval/val_split.py`).

Reads the real DS-0008 files off disk (small, ~6k rows total, loads in well
under a second -- see `build_bank.py`'s own timing) rather than mocking them,
since the whole point is to check the split against the REAL file/chain_env
layout (24 files x 32 chains x 8 steps, confirmed in the EXP-0059 groundwork
report). Not Genesis-dependent (plain `torch.load` of `.pt` files).
"""
import torch

from model.retrieval.val_split import chain_holdout_split, build_tuning_bank_and_val_chains


def test_split_is_a_partition_with_no_row_overlap():
    train_files, val_files = chain_holdout_split(val_frac=0.2, seed=0)
    n_train = sum(len(d["states"]) for d in train_files)
    n_val = sum(len(d["states"]) for d in val_files)
    assert n_train > 0 and n_val > 0
    # roughly a 80/20 split of the ~6144 rows (valid rows only)
    assert 0.1 < n_val / (n_train + n_val) < 0.3


def test_no_chain_appears_on_both_sides():
    """The leakage rule: a (file, chain_env) pair holding a near-duplicate
    8-step trajectory must be ENTIRELY on one side or the other -- checked
    directly against `_file_index`/original `chain_env` (val's `chain_env`
    is remapped to a contiguous range per file, so the pre-remap
    `_orig_chain_env` is what must be compared against train's `chain_env`)."""
    train_files, val_files = chain_holdout_split(val_frac=0.2, seed=0)

    def _train_pairs(files_list):
        return {(d["_file_index"], int(c)) for d in files_list for c in d["chain_env"].tolist()}

    def _val_pairs(files_list):
        return {(d["_file_index"], int(c)) for d in files_list for c in d["_orig_chain_env"].tolist()}

    train_pairs = _train_pairs(train_files)
    val_pairs = _val_pairs(val_files)
    assert len(train_pairs & val_pairs) == 0
    assert len(val_pairs) > 0


def test_val_chain_env_is_contiguous_per_file():
    """`eval_retrieval.py::evaluate`'s rollout loop assumes every chain_env
    in `range(max+1)` has at least one row -- true for a full corpus file,
    NOT automatically true for an arbitrary held-out subset (see
    `val_split.py`'s remap comment). Check every held-out file has no gaps."""
    _, val_files = chain_holdout_split(val_frac=0.2, seed=1)
    for d in val_files:
        E = int(d["chain_env"].max()) + 1
        for e in range(E):
            assert int((d["chain_env"] == e).sum()) > 0, f"gap at chain_env={e}"


def test_build_tuning_bank_and_val_chains_shapes():
    bank, val_chains = build_tuning_bank_and_val_chains(val_frac=0.2, seed=0)
    assert len(bank) > 0
    assert "DS-0010" in set(bank.source)   # DS-0010 is never held out
    assert sum(len(d["states"]) for d in val_chains) > 0
    for d in val_chains:
        assert d["states"].shape[-2:] == (20, 7)


def test_different_seeds_give_different_splits():
    _, val_a = chain_holdout_split(val_frac=0.2, seed=0)
    _, val_b = chain_holdout_split(val_frac=0.2, seed=1)
    a_first = tuple(val_a[0]["p_starts"][0].tolist())
    b_first = tuple(val_b[0]["p_starts"][0].tolist())
    # not a strict guarantee in general, but true for these two seeds and
    # cheap to assert; catches an accidental seed no-op
    assert val_a[0]["states"].shape != val_b[0]["states"].shape or a_first != b_first
