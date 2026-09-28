"""Split narrow_l20_n20 corpora into a CLEAN copy + archive_removed/ (EXP-0059 task 3,
2026-09-28). Genesis-free, CPU-only.

Never modifies an original payload: writes a sibling `<dir>_clean/` (same per-file schema,
same chunk indices, config yaml copied alongside for chains) and `<dir>/archive_removed/`
(removed rows, same schema, + a `reason` column). Structure rules (coordinator's brief):

  - independent-transition sets (DS-0008 train, DS-0015 train_v2): drop bad ROWS individually,
    keep chain_env/chain_step for provenance.
  - pool sets (DS-0009 test_pools, DS-0011 val_pools): drop bad CANDIDATES from their pools
    (same per-row filter as above -- a pool is just rows sharing a pool_idx).
  - chain/sequence sets scored as ROLLOUTS (DS-0009 test_chains, DS-0016 test_chains_v2): a
    (chunk, chain_env) sequence with ANY bad step is removed WHOLE, archived whole -- a later
    step's start state came from the earlier (possibly illegal/null) step's outcome, so a
    kept later step would be conditioned on a removed one.

"bad" criterion:
  - OLD sets (pre-ISS-010-fix): illegal (`<stem>_legality.pt`'s `illegal_0mm`) OR null
    (`<stem>_nullflag.pt`'s `is_null`) -- both already computed and saved next to every source
    file by `audit_tool_placement.py` / `flag_null_transitions.py`.
  - FRESH sets (post-fix, DS-0015/DS-0016): illegal is already verified 0% and gap_out_of_window
    already verified 0% at full scale (see clean_data_status.md), so the only flagged rows left
    are `valid == False` (push length/perpendicularity check failed and could not be redrawn --
    verified to already subsume every null row in these sets).

Usage: python -u experiments/EXP-0059-retrieval-transition-model/code/split_clean_archive.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
D = REPO / "Genesis/data/narrow_l20_n20"


def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _bad_mask_old(data_path: Path):
    """OLD (pre-fix) sets: illegal OR null, from the sidecar flag files. Returns
    (bad: bool tensor, reason: list[str])."""
    leg_path = data_path.with_name(data_path.stem + "_legality.pt")
    null_path = data_path.with_name(data_path.stem + "_nullflag.pt")
    illegal = torch.load(leg_path, map_location="cpu", weights_only=False)["illegal_0mm"].bool()
    is_null = torch.load(null_path, map_location="cpu", weights_only=False)["is_null"].bool()
    bad = illegal | is_null
    reason = []
    for ill, nul in zip(illegal.tolist(), is_null.tolist()):
        if ill and nul:
            reason.append("illegal_and_null")
        elif ill:
            reason.append("illegal")
        elif nul:
            reason.append("null")
        else:
            reason.append("")
    return bad, reason


def _bad_mask_fresh(data):
    """FRESH (post-fix) sets: valid == False (already verified to subsume null; illegal and
    gap_out_of_window are both 0 at full scale, see clean_data_status.md)."""
    bad = ~data["valid"].bool()
    reason = ["invalid_redraw_exhausted" if b else "" for b in bad.tolist()]
    return bad, reason


def _filter_rows(data: dict, keep_idx: torch.Tensor):
    out = {}
    n = len(data[next(k for k in data if torch.is_tensor(data[k]))])
    for k, v in data.items():
        if torch.is_tensor(v) and v.shape[0] == n:
            out[k] = v[keep_idx]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in keep_idx.tolist()]
        else:
            out[k] = v
    return out


def split_independent_or_pool(name, glob_pattern, config_glob, clean_dir, archive_dir, is_fresh):
    """Drop bad rows individually -- covers 'independent training set' and 'pool' kinds
    alike (a pool is just rows sharing pool_idx; dropping bad rows IS dropping bad
    candidates from the pool)."""
    files = sorted(f for f in glob.glob(glob_pattern) if "_legality" not in f and "_nullflag" not in f)
    total = 0
    n_removed = 0
    clean_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        fp = Path(f)
        data = torch.load(fp, map_location="cpu", weights_only=False)
        n = len(data["valid"]) if "valid" in data else len(data["p_starts"])
        bad, reason = (_bad_mask_fresh(data) if is_fresh else _bad_mask_old(fp))
        keep_idx = torch.nonzero(~bad).flatten()
        drop_idx = torch.nonzero(bad).flatten()
        total += n
        n_removed += len(drop_idx)

        clean = _filter_rows(data, keep_idx)
        _atomic_save(clean, clean_dir / fp.name)

        if len(drop_idx):
            archived = _filter_rows(data, drop_idx)
            archived["reason"] = [reason[i] for i in drop_idx.tolist()]
            archived["source_file"] = str(fp)
            archived["source_row"] = drop_idx
            _atomic_save(archived, archive_dir / (fp.stem + "_removed.pt"))

        # config yaml sibling, if present (chains-shaped sets), copied unchanged so the
        # clean dir is directly a PileSweepData training directory.
        cfg = fp.with_name(fp.stem.replace("_data", "") + "_config.yaml")
        if cfg.exists():
            shutil.copy2(cfg, clean_dir / cfg.name)

    print(f"{name}: {total} rows, {n_removed} removed ({n_removed/max(1,total):.4f}), "
         f"clean={total-n_removed} -> {clean_dir}")
    return dict(total=total, removed=n_removed, clean=total - n_removed,
               clean_dir=str(clean_dir), archive_dir=str(archive_dir))


def split_chain_sequences(name, glob_pattern, clean_dir, archive_dir, is_fresh):
    """A (chunk-file, chain_env) sequence with ANY bad step is removed WHOLE."""
    files = sorted(f for f in glob.glob(glob_pattern) if "_legality" not in f and "_nullflag" not in f)
    total = 0
    n_removed = 0
    n_seq_removed = 0
    n_seq_total = 0
    clean_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        fp = Path(f)
        data = torch.load(fp, map_location="cpu", weights_only=False)
        n = len(data["valid"])
        bad, own_reason = (_bad_mask_fresh(data) if is_fresh else _bad_mask_old(fp))
        chain_env = data["chain_env"]
        envs = torch.unique(chain_env)
        bad_env = set()
        for e in envs.tolist():
            if bool(bad[chain_env == e].any()):
                bad_env.add(e)
        seq_bad_row = torch.tensor([int(ce) in bad_env for ce in chain_env.tolist()], dtype=torch.bool)
        keep_idx = torch.nonzero(~seq_bad_row).flatten()
        drop_idx = torch.nonzero(seq_bad_row).flatten()
        total += n
        n_removed += len(drop_idx)
        n_seq_total += len(envs)
        n_seq_removed += len(bad_env)

        clean = _filter_rows(data, keep_idx)
        _atomic_save(clean, clean_dir / fp.name)

        if len(drop_idx):
            archived = _filter_rows(data, drop_idx)
            reason = []
            for i in drop_idx.tolist():
                reason.append(own_reason[i] if own_reason[i] else "sequence_contains_bad_step")
            archived["reason"] = reason
            archived["source_file"] = str(fp)
            archived["source_row"] = drop_idx
            _atomic_save(archived, archive_dir / (fp.stem + "_removed.pt"))

        cfg = fp.with_name(fp.stem.replace("_data", "") + "_config.yaml")
        if cfg.exists():
            shutil.copy2(cfg, clean_dir / cfg.name)

    print(f"{name}: {total} rows / {n_seq_total} sequences, {n_removed} rows removed "
         f"({n_seq_removed} whole sequences, {n_seq_removed/max(1,n_seq_total):.4f}), "
         f"clean={total-n_removed} -> {clean_dir}")
    return dict(total=total, removed=n_removed, clean=total - n_removed,
               sequences_total=n_seq_total, sequences_removed=n_seq_removed,
               clean_dir=str(clean_dir), archive_dir=str(archive_dir))


def main():
    results = {}

    # ---- OLD sets (pre-fix): bad = illegal OR null ----
    results["DS-0008_train"] = split_independent_or_pool(
        "DS-0008_train", str(D / "train/_*_data.pt"), None,
        D / "train_clean", D / "train/archive_removed", is_fresh=False)

    results["DS-0009_test_pools"] = split_independent_or_pool(
        "DS-0009_test_pools", str(D / "test_pools/pools_*.pt"), None,
        D / "test_pools_clean", D / "test_pools/archive_removed", is_fresh=False)

    results["DS-0009_test_chains"] = split_chain_sequences(
        "DS-0009_test_chains", str(D / "test_chains/_*_data.pt"),
        D / "test_chains_clean", D / "test_chains/archive_removed", is_fresh=False)

    results["DS-0011_val_pools"] = split_independent_or_pool(
        "DS-0011_val_pools", str(D / "val_pools/pools_*.pt"), None,
        D / "val_pools_clean", D / "val_pools/archive_removed", is_fresh=False)

    # ---- FRESH sets (post-fix): bad = valid == False (subsumes null; illegal/gap_out_of_window already 0) ----
    results["DS-0015_train_v2"] = split_independent_or_pool(
        "DS-0015_train_v2", str(D / "train_v2/_*_data.pt"), None,
        D / "train_v2_clean", D / "train_v2/archive_removed", is_fresh=True)

    results["DS-0016_test_chains_v2"] = split_chain_sequences(
        "DS-0016_test_chains_v2", str(D / "test_chains_v2/_*_data.pt"),
        D / "test_chains_v2_clean", D / "test_chains_v2/archive_removed", is_fresh=True)

    out = REPO / "experiments/EXP-0059-retrieval-transition-model/results/split_clean_archive.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
