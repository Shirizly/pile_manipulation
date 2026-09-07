"""Rebuild a multi-step slate cell's manifest.json after an interrupted run.

`Genesis/same_state_slate_collection.py` writes `manifest.json` once, at the
very end of a whole run, so a cell that was killed mid-collection leaves batch
files that no loader can group into (slate, step) -- the on-disk schema does not
carry a step index (see `collect_data_samples`, which flattens n_samples x
n_envs into one file).

The mapping is deterministic and was verified against two completed cells
(`n20_L20mm`, `n20_L40mm`): the driver calls `collect_data_samples` once per
step per slate, in order, and the auto-incrementing batch counter therefore
satisfies

    batch_idx == slate_idx * n_steps + step_idx

with `state_library_index` constant across a slate's steps. This script applies
that mapping to whatever complete slates are on disk, DROPS a trailing partial
slate (a sequence missing a step is not a usable sequence), and writes a
manifest carrying the same fields the driver writes plus `rebuilt: true` and the
count of dropped batches, so a reader can tell a salvaged cell from a clean one.

Usage:
    PYTHONPATH=. python scripts/probes/rebuild_slate_manifest.py \
        Genesis/data/slates_multistep/n50_L20mm --n-steps 3 [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--apply", action="store_true",
                    help="write manifest.json (default: report only)")
    args = ap.parse_args()

    root = Path(args.cell)
    idxs = sorted(int(re.match(r"_(\d+)_data\.pt$", p.name).group(1))
                  for p in root.glob("*_data.pt"))
    if idxs != list(range(len(idxs))):
        raise SystemExit(f"batch indices are not contiguous from 0: {idxs[:5]}...")

    n_complete = len(idxs) // args.n_steps
    dropped = len(idxs) - n_complete * args.n_steps
    print(f"{root}: {len(idxs)} batches -> {n_complete} complete slates "
          f"x {args.n_steps} steps, {dropped} trailing batch(es) dropped")
    if n_complete == 0:
        raise SystemExit("no complete slate; nothing to salvage")

    batches = []
    for b in range(n_complete * args.n_steps):
        d = torch.load(root / f"_{b}_data.pt", map_location="cpu", weights_only=False)
        slate, step = divmod(b, args.n_steps)
        batches.append({
            "batch_idx": b,
            "slate_idx": slate,
            "step_idx": step,
            "env_count": int(d["p_starts"].shape[0]),
            "state_library_index": None,   # not recoverable from the data files
            "unresolved_after_resampling": None,
        })

    cfg = yaml.full_load((root / "_0_config.yaml").read_text())
    dc = cfg.get("data_collection", {})
    man = {
        "rebuilt": True,
        "rebuilt_by": "scripts/probes/rebuild_slate_manifest.py",
        "rebuilt_note": (
            "Collection was interrupted; this manifest was reconstructed from the "
            "deterministic batch_idx == slate_idx * n_steps + step_idx mapping. "
            "state_library_index and unresolved_after_resampling are unrecoverable "
            "and are null. Trailing incomplete slates were dropped."),
        "dropped_trailing_batches": dropped,
        "n_steps": args.n_steps,
        "n_slates": n_complete,
        "n_cubes": dc.get("n_cubes"),
        "size": dc.get("cube_size"),
        "push_length": dc.get("push_length"),
        "n_envs": dc.get("n_envs_per_slate"),
        "batches": batches,
    }
    if args.apply:
        (root / "manifest.json").write_text(json.dumps(man, indent=2))
        print(f"wrote {root / 'manifest.json'}")
    else:
        print("(dry run; pass --apply to write)")


if __name__ == "__main__":
    main()
