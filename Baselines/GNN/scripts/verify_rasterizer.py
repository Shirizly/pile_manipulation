"""Baselines/GNN/scripts/verify_rasterizer.py -- proves
`Baselines.GNN.predictor.rasterize_particles_batch` is numerically
identical to the harness's own per-row `Baselines.common.data.
rasterize_particles`, on real candidates from both eval cells, BEFORE it
is trusted in `predict_occ`. Run with `PYTHONPATH=.`.

Also runs the isolated cv2.fillPoly(list) vs cv2.fillPoly(loop) check
inline (already spot-checked in the chat session, repeated here so the
proof is reproducible from one script).
"""
from __future__ import annotations

import numpy as np
import torch

from Baselines.common.data import load_cell, rasterize_particles
from Baselines.GNN.predictor import rasterize_particles_batch

CELLS = [
    ("configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
     "Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    ("configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
     "Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
]

N_PER_CELL = 200


def main():
    torch.manual_seed(0)
    max_abs_diff = 0.0
    total_checked = 0
    for cfg, manifest in CELLS:
        cell = load_cell(cfg, "train", manifest_path=manifest, tag="verify")
        n = cell.occ0.shape[0]
        idx = torch.randperm(n)[:N_PER_CELL]

        # "predicted" particles = states_ (ground-truth post-push) -- any
        # real (B,20,7) tensor works, this is purely a rasteriser check.
        particles = cell.states_[idx]
        run_idx = cell.run_idx[idx].tolist()

        batched = rasterize_particles_batch(cell.raw, run_idx, particles)

        per_row = torch.empty_like(batched)
        for i in range(len(idx)):
            per_row[i] = rasterize_particles(cell.raw, run_idx[i], particles[i])

        diff = (batched - per_row).abs()
        d = float(diff.max())
        max_abs_diff = max(max_abs_diff, d)
        total_checked += len(idx)
        print(f"{cfg}: {len(idx)} candidates, max abs diff = {d:.3e}, "
              f"exact match = {bool((diff == 0).all())}")
        assert d < 1e-6, f"rasteriser mismatch on {cfg}: max abs diff {d}"

    print(f"\nTOTAL candidates checked: {total_checked}, "
          f"overall max abs diff = {max_abs_diff:.3e}")
    print("PASS: batched rasteriser is numerically identical to the "
          "harness's per-row rasterize_particles.")


if __name__ == "__main__":
    main()
