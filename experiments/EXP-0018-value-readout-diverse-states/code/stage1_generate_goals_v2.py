"""Stage 1 (EXP-0018) -- extend EXP-0015's stage3a goal generator to ~2000 goals.

Superset of EXP-0015's shape families (rectangles + letters T/A/O/S/G at random
rotation) plus ellipses, two-blob (two disks) and L/E/H/C letters, with a wider
area range. Every goal is stored BOTH as a mask and as K legal, non-penetrating
20-cube material configurations (`Baselines/common/goal_configs.mask_to_configuration`).

Writes experiments/temp/goal-states/dataset_v2.pt -- does NOT touch dataset.pt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate as nd_rotate

REPO = Path(__file__).resolve().parents[3]
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
SIDE = BOUNDS["x_max"] - BOUNDS["x_min"]
K_CONFIG_SAMPLES = 3
SEED = 2026

N_RECT = 800
LETTERS = ["T", "A", "O", "S", "G", "L", "E", "H", "C"]
N_LETTERS_PER = 90          # 810
N_ELLIPSE = 250
N_TWOBLOB = 140             # total 2000


def _grid_xy():
    ys = np.linspace(BOUNDS["y_min"], BOUNDS["y_max"], GRID, endpoint=False) + \
        (BOUNDS["y_max"] - BOUNDS["y_min"]) / GRID / 2
    xs = np.linspace(BOUNDS["x_min"], BOUNDS["x_max"], GRID, endpoint=False) + \
        (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID / 2
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return gy, gx


GY, GX = _grid_xy()


def _local(cx, cy, angle):
    dx = GX - cx
    dy = GY - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    return ca * dx - sa * dy, sa * dx + ca * dy


def _centre(rng, extent):
    """Centre sampled so the shape stays mostly inside the tray."""
    lim = max(SIDE / 2 - 0.35 * extent, 0.0)
    return float(rng.uniform(-lim, lim)), float(rng.uniform(-lim, lim))


def sample_rect(rng):
    frac = float(rng.uniform(0.06, 0.45))          # wider than EXP-0015's 0.15-0.50 beta
    area = frac * SIDE ** 2
    aspect = float(np.exp(rng.uniform(np.log(0.3), np.log(3.3))))
    w = min(float(np.sqrt(area * aspect)), SIDE * 0.95)
    h = min(float(np.sqrt(area / aspect)), SIDE * 0.95)
    ang = float(rng.uniform(0, 2 * np.pi))
    cx, cy = _centre(rng, max(w, h))
    lx, ly = _local(cx, cy, ang)
    m = (np.abs(lx) <= w / 2) & (np.abs(ly) <= h / 2)
    return m, {"type": "rectangle", "area_frac_nominal": frac, "w_m": w, "h_m": h,
               "angle_rad": ang, "cx_m": cx, "cy_m": cy}


def sample_ellipse(rng):
    frac = float(rng.uniform(0.05, 0.40))
    area = frac * SIDE ** 2
    aspect = float(np.exp(rng.uniform(np.log(0.3), np.log(3.3))))
    a = float(np.sqrt(area * aspect / np.pi))
    b = float(np.sqrt(area / aspect / np.pi))
    ang = float(rng.uniform(0, 2 * np.pi))
    cx, cy = _centre(rng, 2 * max(a, b))
    lx, ly = _local(cx, cy, ang)
    m = (lx / a) ** 2 + (ly / b) ** 2 <= 1.0
    return m, {"type": "ellipse", "area_frac_nominal": frac, "a_m": a, "b_m": b,
               "angle_rad": ang, "cx_m": cx, "cy_m": cy}


def sample_twoblob(rng):
    r1 = float(rng.uniform(0.09, 0.22)) * SIDE
    r2 = float(rng.uniform(0.09, 0.22)) * SIDE
    sep = float(rng.uniform(1.0, 2.6)) * (r1 + r2) / 2
    ang = float(rng.uniform(0, 2 * np.pi))
    cx, cy = _centre(rng, sep + r1 + r2)
    ox, oy = np.cos(ang) * sep / 2, np.sin(ang) * sep / 2
    m = (((GX - cx - ox) ** 2 + (GY - cy - oy) ** 2) <= r1 ** 2) | \
        (((GX - cx + ox) ** 2 + (GY - cy + oy) ** 2) <= r2 ** 2)
    return m, {"type": "twoblob", "r1_m": r1, "r2_m": r2, "sep_m": sep,
               "angle_rad": ang, "cx_m": cx, "cy_m": cy}


def sample_letter(rng, letter):
    base = letter_mask(letter, GRID, GRID).astype(np.float32)
    deg = float(rng.uniform(0, 360))
    rot = nd_rotate(base, deg, reshape=False, order=1, mode="constant", cval=0.0)
    scale = float(rng.uniform(0.75, 1.0))
    m = rot > 0.5
    if scale < 0.999:                    # shrink toward the centre by zooming out
        from scipy.ndimage import zoom
        z = zoom(rot, scale, order=1)
        out = np.zeros((GRID, GRID), np.float32)
        o = (GRID - z.shape[0]) // 2
        out[o:o + z.shape[0], o:o + z.shape[1]] = z[:GRID, :GRID]
        m = out > 0.5
    return m, {"type": "letter", "letter": letter, "angle_deg": deg, "scale": scale}


def main():
    rng = np.random.default_rng(SEED)
    masks, metas = [], []
    t0 = time.time()
    for _ in range(N_RECT):
        m, meta = sample_rect(rng); masks.append(m); metas.append(meta)
    for letter in LETTERS:
        for _ in range(N_LETTERS_PER):
            m, meta = sample_letter(rng, letter); masks.append(m); metas.append(meta)
    for _ in range(N_ELLIPSE):
        m, meta = sample_ellipse(rng); masks.append(m); metas.append(meta)
    for _ in range(N_TWOBLOB):
        m, meta = sample_twoblob(rng); masks.append(m); metas.append(meta)

    # drop degenerate (empty / whole-tray) masks
    keep = [i for i, m in enumerate(masks) if 12 <= int(m.sum()) <= GRID * GRID - 12]
    dropped = len(masks) - len(keep)
    masks = [masks[i] for i in keep]; metas = [metas[i] for i in keep]
    print(f"generated {len(masks)} masks ({dropped} degenerate dropped) in {time.time()-t0:.1f}s", flush=True)
    areas = np.array([m.sum() for m in masks])
    print(f"mask area px: min={areas.min()} max={areas.max()} mean={areas.mean():.1f}", flush=True)

    t0 = time.time()
    N = len(masks)
    configs = np.zeros((N, K_CONFIG_SAMPLES, 20, 7), dtype=np.float64)
    methods = []
    for gi, m in enumerate(masks):
        gm = []
        for k in range(K_CONFIG_SAMPLES):
            res = mask_to_configuration(m, n_objects=20, seed=SEED * 100000 + gi * 100 + k)
            assert_no_penetration(res.poses[:, :2])
            configs[gi, k] = res.poses
            gm.append(res.method)
        methods.append(gm)
        if gi % 500 == 0:
            print(f"  cfg {gi}/{N} ({time.time()-t0:.1f}s)", flush=True)
    print(f"{N*K_CONFIG_SAMPLES} configurations, ALL passed assert_no_penetration, "
          f"{time.time()-t0:.1f}s", flush=True)

    torch.save({
        "masks": torch.from_numpy(np.stack(masks)),
        "configs": torch.from_numpy(configs),
        "methods": methods, "metadata": metas,
        "grid": GRID, "bounds": BOUNDS, "k_config_samples": K_CONFIG_SAMPLES,
        "letters": LETTERS, "seed": SEED, "provenance": git_provenance(),
        "families": {"rectangle": N_RECT, "letter": len(LETTERS) * N_LETTERS_PER,
                     "ellipse": N_ELLIPSE, "twoblob": N_TWOBLOB},
        "source": "EXP-0018 stage1_generate_goals_v2.py (superset of EXP-0015 stage3a)",
    }, OUT_DIR / "dataset_v2.pt")
    print(f"saved {OUT_DIR/'dataset_v2.pt'}", flush=True)


if __name__ == "__main__":
    main()
