"""EXP-B multistep cells: deterministic slate-level 30/20 train/eval split.

`Genesis/data/slates_multistep/<cell>/` holds 50 slates x 3 steps x 128 envs,
one file per (slate, step) batch (`batch_idx = slate_idx * n_steps + step_idx`,
per the manifest). `Genesis/training/dataset.py::PileSweepData` splits at
FILE granularity (physics-group hash bucketing), which here would split
individual (slate, step) batches across train/eval -- since every file in one
of these cells shares one nominal physics key, that is exactly the leak the
programme forbids (`docs/plan_selection_pressure_validation.md` design:
"Train/eval split: slate-level ... One slate is one initial state, so a
slate-level split leaks nothing").

This script does the split OUTSIDE that mechanism, at the one granularity that
is safe: whole slates (all 3 of their step files move together). It builds two
sibling directories of symlinks -- `<cell>_train/` (30 slates x 3 steps = 90
files) and `<cell>_eval/` (20 slates x 3 steps = 60 files) -- so every existing
dataset-config / PileSweepData / build_dataset code path can load them
unchanged with val_pct=0/test_pct=0/split="train" (the
`genesis_cube_spectrum_n20_slates_all.yaml` pattern), with no new loader code.

Split rule (fixed, documented, reused identically for every cell): sort slate
indices 0..49, permute with `numpy.random.default_rng(0)`, first 30 (sorted)
-> train, remaining 20 (sorted) -> eval. Symlinks, not copies -- these files
are read-only inputs and a copy would cost ~20 MB x2 for nothing.

Usage
-----
    PYTHONPATH=. python scripts/probes/prepare_slate_multistep_split.py \
        Genesis/data/slates_multistep/n20_L10mm
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SEED = 0
N_TRAIN = 30


def split_slate_indices(n_slates: int, seed: int = SEED, n_train: int = N_TRAIN):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_slates)
    train_idx = sorted(int(x) for x in perm[:n_train])
    eval_idx = sorted(int(x) for x in perm[n_train:])
    return train_idx, eval_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_root")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    root = Path(args.cell_root)
    manifest = json.loads((root / "manifest.json").read_text())
    n_slates = manifest["n_states_collected"]
    n_steps = manifest["n_steps"]
    batches = manifest["batches"]

    train_idx, eval_idx = split_slate_indices(n_slates, args.seed, args.n_train)
    train_set, eval_set = set(train_idx), set(eval_idx)
    assert train_set.isdisjoint(eval_set)
    assert train_set | eval_set == set(range(n_slates))
    print(f"{root.name}: {n_slates} slates -> {len(train_idx)} train / "
          f"{len(eval_idx)} eval (seed={args.seed})")
    print(f"  train slate idx: {train_idx}")
    print(f"  eval  slate idx: {eval_idx}")

    by_slate_step = {(b["slate_idx"], b["step_idx"]): b["batch_idx"] for b in batches}

    for tag, idx_set in (("train", train_set), ("eval", eval_set)):
        out_dir = root.parent / f"{root.name}_{tag}"
        out_dir.mkdir(exist_ok=True)
        n_linked = 0
        for slate in sorted(idx_set):
            for step in range(n_steps):
                bidx = by_slate_step[(slate, step)]
                for suffix in ("_data.pt", "_config.yaml"):
                    src = root / f"_{bidx}{suffix}"
                    dst = out_dir / f"_{bidx}{suffix}"
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src.resolve(), dst)
                n_linked += 1
        manifest_out = {
            "source_cell": str(root), "seed": args.seed, "n_train": args.n_train,
            "n_slates": n_slates, "n_steps": n_steps,
            "train_slate_idx": train_idx, "eval_slate_idx": eval_idx,
            "split": tag, "slate_idx": sorted(idx_set),
            "batch_idx": sorted(by_slate_step[(s, st)] for s in idx_set
                                 for st in range(n_steps)),
        }
        (out_dir / "split_manifest.json").write_text(json.dumps(manifest_out, indent=2))
        print(f"  {out_dir}: {n_linked} batches symlinked "
              f"({n_linked * manifest['batches'][0]['env_count']} transitions)")


if __name__ == "__main__":
    main()
