"""C-006 re-run on corrected grids (post aac084e3): profile the mean
canonical-frame delta <I_{k+1} - I_k> along the push axis, on a scattered and
a piled dataset, through ONE code path (load_transition_fields) so the two
datasets differ only in the data, not the rasteriser (see EXP-0001/EXP-0002 on
why that matters).

Canonical-frame axis convention (verified against transforms/functional.py):
push_frame_transform makes canonical +x the push direction and affine_grid/
grid_sample place x on the OUTPUT's last dim (columns) and y on the
second-to-last (rows). So profiling "along the push axis" means averaging a
row-band around the centre row, then reading off by COLUMN.

No fitting here -- this is a descriptive statistic of the data itself, so the
whole loaded set is used (no train/test split needed for a mean).
"""
import argparse
import torch

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import canonicalise, actions_to_pixels

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--band-px", type=float, default=4.0)
ap.add_argument("--n-folds", type=int, default=4)
ap.add_argument("--max-episodes", type=int, default=None,
               help="cap episode FILES loaded, for a bounded-time rerun -- "
                    "particles_to_occupancy's footprint_radius path is a "
                    "per-batch-item Python loop (O(B*N*res^2)), which made the "
                    "unbounded run take ~11.5 min wall for two datasets "
                    "(logged as an unrelated finding).")
a = ap.parse_args()
H = W = 64


def profile(name, glob, cube_size, min_push_mm, min_grains):
    print(f"  loading {name} ...", flush=True)
    o0, o1, act, ep, _, _ = load_transition_fields(
        glob, 64, a.blur, "mean", min_push_mm, "cpu",
        view="mask", min_grains=min_grains, cube_size=cube_size,
        max_episodes=a.max_episodes)
    print(f"  loaded {o0.shape[0]} transitions", flush=True)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    Y0 = canonicalise(o0, s_px, e_px, a.res, a.crop)   # (N, res, res)
    Y1 = canonicalise(o1, s_px, e_px, a.res, a.crop)
    delta = Y1 - Y0

    center = (a.res - 1) / 2.0
    rows = torch.arange(a.res, dtype=torch.float32)
    band = (rows - center).abs() <= a.band_px          # perpendicular band
    prof_full = delta[:, band, :].mean(dim=(0, 1))     # (res,) indexed by col

    # noise floor: episode-level folds
    eps = ep.unique()
    g = torch.Generator().manual_seed(0)
    perm = eps[torch.randperm(len(eps), generator=g)]
    folds = torch.chunk(perm, a.n_folds)
    fold_profiles = []
    for f in folds:
        m = torch.tensor([int(e) in set(f.tolist()) for e in ep])
        if int(m.sum()) < 5:
            continue
        fold_profiles.append(delta[m][:, band, :].mean(dim=(0, 1)))
    fp = torch.stack(fold_profiles)                    # (nfolds, res)
    fold_sd = fp.std(dim=0)

    print(f"\n=== {name}  (N={o0.shape[0]} transitions, {len(eps)} episodes, "
          f"res={a.res} crop={a.crop} blur={a.blur}, band=+/-{a.band_px}px) ===")
    print(f"{'col offset':>12s} {'mean delta':>12s} {'fold sd':>10s}")
    for c in range(a.res):
        off = c - center
        print(f"{off:12.1f} {float(prof_full[c]):12.5f} {float(fold_sd[c]):10.5f}")
    peak_dep = int(prof_full.argmin())
    peak_depo = int(prof_full.argmax())
    print(f"peak depletion  at col offset {peak_dep - center:+.1f}: "
          f"{float(prof_full[peak_dep]):.5f}  (fold sd {float(fold_sd[peak_dep]):.5f})")
    print(f"peak deposition at col offset {peak_depo - center:+.1f}: "
          f"{float(prof_full[peak_depo]):.5f}  (fold sd {float(fold_sd[peak_depo]):.5f})")
    return prof_full, fold_sd


profile("SCATTERED monolayer (L040, n50)", "Genesis/data/foresight/L040/**/*_data.pt",
        0.005, 39.0, 1.0)
profile("PILED cubes (n20, 2-layer heap)", "Genesis/data/cube_spectrum/n20/*_data.pt",
        0.005, 19.9, 1.0)
