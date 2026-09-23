"""Stage 1 check: generate goal configurations, assert no-penetration, and
cross-check rasterised pixel mass against real corpus states."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from transforms.functional import particles_to_occupancy               # noqa: E402
from Baselines.common.goals import quadrant_mask, letter_mask            # noqa: E402
from Baselines.common.goal_configs import (                              # noqa: E402
    mask_to_configuration, assert_no_penetration, DEFAULT_BOUNDS, CUBE_SIZE,
)
from Genesis.binned_slate_dataset import BinnedSlateCorpus                # noqa: E402

GRID = 64
PITCH_PX = (DEFAULT_BOUNDS["x_max"] - DEFAULT_BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH_PX

GOALS = {
    "quadrant0": quadrant_mask(GRID, GRID, 0),
    "quadrant1": quadrant_mask(GRID, GRID, 1),
    "quadrant2": quadrant_mask(GRID, GRID, 2),
    "quadrant3": quadrant_mask(GRID, GRID, 3),
    "letter_T": letter_mask("T", GRID, GRID),
    "letter_O": letter_mask("O", GRID, GRID),
}


def occ_of(poses):
    t = torch.from_numpy(poses[None, :, :3]).float()
    return particles_to_occupancy(t, DEFAULT_BOUNDS, (GRID, GRID), footprint_radius=RADIUS)[0]


def main():
    print("=== generating configurations, N_SAMPLES=10 per goal ===", flush=True)
    N_SAMPLES = 10
    masses = {}
    for name, mask in GOALS.items():
        ms = []
        methods = set()
        for s in range(N_SAMPLES):
            res = mask_to_configuration(mask, n_objects=20, seed=1000 + s)
            assert_no_penetration(res.poses[:, :2])  # re-assert independently
            occ = occ_of(res.poses)
            ms.append(float(occ.sum()))
            methods.add(res.method)
        masses[name] = ms
        print(f"{name:10s} method={methods} pixel_mass mean={np.mean(ms):.1f} "
              f"std={np.std(ms):.1f} (raw mask pixel count={int(mask.sum())})", flush=True)

    print("\n=== cross-check vs real corpus state pixel mass ===", flush=True)
    corpus = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"))
    rows = corpus.step(0)
    states_ = rows.states_[:500].float()
    occ_real = particles_to_occupancy(states_[..., :3], DEFAULT_BOUNDS, (GRID, GRID),
                                       footprint_radius=RADIUS)
    real_mass = occ_real.sum(dim=(-2, -1)).numpy()
    print(f"real states (n={len(real_mass)}): pixel_mass mean={real_mass.mean():.1f} "
          f"std={real_mass.std():.1f} min={real_mass.min():.1f} max={real_mass.max():.1f}", flush=True)
    all_gen_mass = np.concatenate(list(masses.values()))
    print(f"generated configs (n={len(all_gen_mass)}): pixel_mass mean={all_gen_mass.mean():.1f} "
          f"std={all_gen_mass.std():.1f}", flush=True)
    print("\nNO-PENETRATION ASSERTIONS: all passed (would have raised otherwise)", flush=True)


if __name__ == "__main__":
    main()
