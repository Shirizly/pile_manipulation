"""Genesis/data/overnight_randlen: deterministic 90/10 train/test split at
FILE (episode) granularity.

Mirrors `scripts/probes/prepare_slate_multistep_split.py`'s convention
(sibling symlink directories, `numpy.random.default_rng(0)`, exact indices
recorded in a manifest) but applied to this corpus's layout: five leaf
directories (spawn-mode x particle-count groups), each holding N independent
episode files `_<i>_data.pt` / `_<i>_config.yaml` (one file = 128 envs x
`samples_per_env` sequential-chain transitions, confirmed by direct
measurement that within a file, row i's outcome state feeds row i+128's start
state -- see the experiment record). A transition-level split would leak
because consecutive rows within a file share a chained pile trajectory
(`episode-split` invariant, docs/experiments/INVARIANTS.md); a file-level
split does not, since every file's 512 rows move together.

Split rule (fixed, per group, so each spawn-mode/count group is represented
in both splits): sort file indices 0..N-1, permute with
`numpy.random.default_rng(0)`, first `round(0.9*N)` (at least N-1) -> train,
remainder -> test. Symlinks, not copies.

Usage
-----
    PYTHONPATH=. python scripts/probes/prepare_randlen_split.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

SEED = 0
TEST_FRAC = 0.10

ROOT = Path("Genesis/data/overnight_randlen")
GROUPS = {
    "mixed_n20": ROOT / "mixed/cube/n20/size0.005/pile0.5_layers4",
    "piled_n20": ROOT / "piled/cube/n20/size0.005/layers4",
    "piled_n50": ROOT / "piled/cube/n50/size0.005/layers4",
    "scattered_n20": ROOT / "scattered/cube/n20/size0.005",
    "scattered_n50": ROOT / "scattered/cube/n50/size0.005",
}

OUT_TRAIN = Path("Genesis/data/overnight_randlen_train")
OUT_TEST = Path("Genesis/data/overnight_randlen_test")


def file_indices(group_dir: Path) -> list[int]:
    idx = []
    for p in glob.glob(str(group_dir / "_*_data.pt")):
        m = re.match(r"_(\d+)_data\.pt$", os.path.basename(p))
        assert m, p
        idx.append(int(m.group(1)))
    return sorted(idx)


def main():
    overall = {"seed": SEED, "test_frac": TEST_FRAC, "groups": {}}
    for name, group_dir in GROUPS.items():
        idx = file_indices(group_dir)
        n = len(idx)
        n_test = max(1, round(TEST_FRAC * n))
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n)
        test_pos = sorted(int(x) for x in perm[:n_test])
        train_pos = sorted(int(x) for x in perm[n_test:])
        test_idx = sorted(idx[p] for p in test_pos)
        train_idx = sorted(idx[p] for p in train_pos)
        assert set(test_idx).isdisjoint(train_idx)
        assert set(test_idx) | set(train_idx) == set(idx)

        print(f"{name}: {n} files -> {len(train_idx)} train / {len(test_idx)} test")
        print(f"  test file idx: {test_idx}")

        for tag, split_idx, out_root in (
            ("train", train_idx, OUT_TRAIN), ("test", test_idx, OUT_TEST),
        ):
            out_dir = out_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            n_linked = 0
            for i in split_idx:
                for suffix in ("_data.pt", "_config.yaml"):
                    src = group_dir / f"_{i}{suffix}"
                    dst = out_dir / f"_{i}{suffix}"
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src.resolve(), dst)
                n_linked += 1
            print(f"  {out_dir}: {n_linked} files symlinked ({tag})")

        overall["groups"][name] = {
            "source": str(group_dir), "n_files": n,
            "train_idx": train_idx, "test_idx": test_idx,
        }

    n_train_total = sum(len(g["train_idx"]) for g in overall["groups"].values())
    n_test_total = sum(len(g["test_idx"]) for g in overall["groups"].values())
    overall["n_train_files_total"] = n_train_total
    overall["n_test_files_total"] = n_test_total
    print(f"\nTOTAL: {n_train_total} train files / {n_test_total} test files "
          f"({n_train_total + n_test_total} total)")

    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    (OUT_TRAIN / "split_manifest.json").write_text(json.dumps(overall, indent=2))
    (OUT_TEST / "split_manifest.json").write_text(json.dumps(overall, indent=2))
    print(f"manifest written to {OUT_TRAIN/'split_manifest.json'} and "
          f"{OUT_TEST/'split_manifest.json'}")


if __name__ == "__main__":
    main()
