"""Stage 3a — generate ~500 goal shapes (masks + legal configurations +
metadata): rectangles (random side lengths, area 15-50% of workspace, biased
toward the smaller end) and letters T/A/O/S/G at random rotation. Saved to
experiments/temp/goal-states/dataset.pt (gitignored, per brief).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate as nd_rotate

REPO = Path(__file__).resolve().parents[3]   # repo root; was parents[4], which resolved OUTSIDE the repo after this script was promoted from experiments/temp/ into experiments/EXP-0015-*/code/
sys.path.insert(0, str(REPO))

from Baselines.common.goals import letter_mask                          # noqa: E402
from Baselines.common.goal_configs import (                             # noqa: E402
    mask_to_configuration, assert_no_penetration, DEFAULT_BOUNDS,
)
from utils import git_provenance                                        # noqa: E402

OUT_DIR = REPO / "experiments/temp/goal-states"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 64
BOUNDS = DEFAULT_BOUNDS
WORKSPACE_SIDE = BOUNDS["x_max"] - BOUNDS["x_min"]  # m
WORKSPACE_AREA = WORKSPACE_SIDE ** 2
PIXEL_AREA = WORKSPACE_AREA / (GRID * GRID)

N_RECT = 300
N_LETTERS_PER = 40  # x5 letters = 200
LETTERS = ["T", "A", "O", "S", "G"]
K_CONFIG_SAMPLES = 3
SEED = 42


def rect_mask(center_xy, w, h, angle_rad):
    """Rotated rectangle mask on the GRID x GRID world-bounds grid, row=y/col=x
    convention (matches Baselines/common/goals.py)."""
    ys = np.linspace(BOUNDS["y_min"], BOUNDS["y_max"], GRID, endpoint=False) + \
        (BOUNDS["y_max"] - BOUNDS["y_min"]) / GRID / 2
    xs = np.linspace(BOUNDS["x_min"], BOUNDS["x_max"], GRID, endpoint=False) + \
        (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID / 2
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    cx, cy = center_xy
    dx = gx - cx; dy = gy - cy
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    return (np.abs(lx) <= w / 2) & (np.abs(ly) <= h / 2)


def sample_rect(rng):
    frac = 0.15 + 0.35 * rng.beta(2, 5)  # biased toward the smaller end of [0.15,0.5]
    area = frac * WORKSPACE_AREA
    aspect = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
    w = float(np.sqrt(area * aspect))
    h = float(np.sqrt(area / aspect))
    w = min(w, WORKSPACE_SIDE * 0.98); h = min(h, WORKSPACE_SIDE * 0.98)
    angle = float(rng.uniform(0, 2 * np.pi))
    margin = 0.5 * max(w, h)
    lo, hi = BOUNDS["x_min"] + 0.15 * margin, BOUNDS["x_max"] - 0.15 * margin
    cx = float(np.clip(rng.uniform(BOUNDS["x_min"], BOUNDS["x_max"]), lo, hi)) \
        if lo < hi else 0.0
    cy = float(np.clip(rng.uniform(BOUNDS["y_min"], BOUNDS["y_max"]), lo, hi)) \
        if lo < hi else 0.0
    mask = rect_mask((cx, cy), w, h, angle)
    return mask, {"type": "rectangle", "area_frac_nominal": frac, "w_m": w, "h_m": h,
                  "angle_rad": angle, "cx_m": cx, "cy_m": cy}


def sample_letter(rng, letter):
    base = letter_mask(letter, GRID, GRID).astype(np.float32)
    angle_deg = float(rng.uniform(0, 360))
    rotated = nd_rotate(base, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    mask = rotated > 0.5
    return mask, {"type": "letter", "letter": letter, "angle_deg": angle_deg}


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    masks = []
    metas = []
    for i in range(N_RECT):
        m, meta = sample_rect(rng)
        masks.append(m); metas.append(meta)
    for letter in LETTERS:
        for i in range(N_LETTERS_PER):
            m, meta = sample_letter(rng, letter)
            masks.append(m); metas.append(meta)
    print(f"generated {len(masks)} raw masks in {time.time()-t0:.1f}s", flush=True)

    areas_px = np.array([m.sum() for m in masks])
    print(f"mask area px: min={areas_px.min()} max={areas_px.max()} "
          f"mean={areas_px.mean():.1f} (grid={GRID*GRID}px total)", flush=True)

    print("generating configurations (K=%d samples each)..." % K_CONFIG_SAMPLES, flush=True)
    all_configs = np.zeros((len(masks), K_CONFIG_SAMPLES, 20, 7), dtype=np.float64)
    methods = []
    n_penetration_checks = 0
    t0 = time.time()
    for gi, m in enumerate(masks):
        gmethods = []
        for k in range(K_CONFIG_SAMPLES):
            res = mask_to_configuration(m, n_objects=20, seed=SEED * 100000 + gi * 100 + k)
            assert_no_penetration(res.poses[:, :2])
            n_penetration_checks += 1
            all_configs[gi, k] = res.poses
            gmethods.append(res.method)
        methods.append(gmethods)
        if gi % 100 == 0:
            print(f"  {gi}/{len(masks)}  ({time.time()-t0:.1f}s elapsed)", flush=True)
    print(f"done in {time.time()-t0:.1f}s; {n_penetration_checks} configurations, "
          f"ALL passed assert_no_penetration", flush=True)

    mask_stack = np.stack(masks)  # (500, H, W) bool
    torch.save({
        "masks": torch.from_numpy(mask_stack),
        "configs": torch.from_numpy(all_configs),   # (N, K, 20, 7)
        "methods": methods,
        "metadata": metas,
        "grid": GRID, "bounds": BOUNDS, "k_config_samples": K_CONFIG_SAMPLES,
        "n_rect": N_RECT, "n_letters_per": N_LETTERS_PER, "letters": LETTERS,
        "seed": SEED, "provenance": git_provenance(),
        "rect_area_frac_distribution": "0.15 + 0.35*Beta(2,5) -- mean~0.25, biased small",
    }, OUT_DIR / "dataset.pt")
    print(f"saved {OUT_DIR/'dataset.pt'}", flush=True)


if __name__ == "__main__":
    main()
