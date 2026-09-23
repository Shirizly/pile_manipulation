"""Validate DS-0003 (synthetic states) against DS-0002 (real distinct states):
rasterise both through ONE fast rasteriser and compare occupancy descriptor
distributions, so the synthetic states plausibly overlap what the dynamics
actually visits (otherwise the encoder spends capacity on regions it will
never see).

Rasteriser: experiments/EXP-0019-*/code/raster.py::rasterise -- fast,
batch-vectorised disk splat, BIT-EXACT against
transforms.functional.particles_to_occupancy (asserted by that experiment's
check_raster.py) and ~80x faster; the project rasteriser loops over the batch
in Python.

Descriptors: dmdc_baseline.py::occupancy_descriptors / descriptor_slices --
const, mass, com(2), moments2(3), dft_real/imag (low-frequency 2D DFT block).

Run:
  /home/alon/anaconda3/envs/pme/bin/python -u datasets/DS-0003-synthetic-states/validate.py
"""
from __future__ import annotations
import glob, json, os, sys, importlib.util

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/alon/Code/pile_manipulation"
sys.path.insert(0, ROOT)

# raster.py lives in an experiment code/ dir -- import by path, don't add its
# whole experiment tree to sys.path (its siblings pull in heavier deps).
raster_path = glob.glob(f"{ROOT}/experiments/EXP-0019-*/code/raster.py")[0]
spec = importlib.util.spec_from_file_location("raster", raster_path)
raster_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raster_mod)
rasterise = raster_mod.rasterise
BASE_RADIUS = raster_mod.BASE_RADIUS
GRID = raster_mod.GRID
BOUNDS = raster_mod.BOUNDS

from dmdc_baseline import occupancy_descriptors, descriptor_slices

DS0002 = f"{ROOT}/datasets/DS-0002-real-distinct-states/data/states.pt"
DS0003 = f"{ROOT}/datasets/DS-0003-synthetic-states/data/states.pt"
OUT_DIR = f"{ROOT}/datasets/DS-0003-synthetic-states/validation"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE = 4000
rng = np.random.default_rng(0)


def sample_and_rasterise(path, n_sample, xyz_key="xyz", n_key="n_objects", per_n=None):
    """per_n: optional dict {n_objects: count} for STRATIFIED sampling, so a
    real corpus dominated by n=20 rows is not compared against a synthetic
    set balanced 1/3-1/3-1/3 -- mass/com/etc. scale with n_objects, so an
    unmatched n_objects mix alone can look like an out-of-distribution shift
    that has nothing to do with the generator's geometry."""
    d = torch.load(path, map_location="cpu")
    n_obj_all = d[n_key]
    if per_n is not None:
        idx_parts = []
        for n, cnt in per_n.items():
            pool = np.nonzero((n_obj_all == n).numpy())[0]
            take = rng.choice(pool, size=min(cnt, len(pool)), replace=False)
            idx_parts.append(take)
        idx = np.concatenate(idx_parts)
    else:
        n_total = n_obj_all.shape[0]
        idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    xyz = d[xyz_key][idx]              # (B, MAX_N, 3)
    n_obj = d[n_key][idx]              # (B,)
    B, MAX_N, _ = xyz.shape
    valid = torch.arange(MAX_N)[None, :] < n_obj[:, None]
    occ = rasterise(xyz[:, :, :2].float(), BASE_RADIUS, grid=GRID, bounds=BOUNDS, valid=valid)
    phi = occupancy_descriptors(occ, n_fourier=8)
    return occ.numpy(), phi.numpy(), n_obj.numpy()


# Stratify BOTH samples to the same n_objects mix (1/3 each of 20/50/100) so
# mass/com/etc., which scale with n_objects, aren't compared across mismatched
# corpus compositions (DS-0002 is dominated by n=20 rows; DS-0003 is balanced).
PER_N = {20: N_SAMPLE // 3, 50: N_SAMPLE // 3, 100: N_SAMPLE // 3}
occ_real, phi_real, n_real = sample_and_rasterise(DS0002, N_SAMPLE, per_n=PER_N)
occ_syn, phi_syn, n_syn = sample_and_rasterise(DS0003, N_SAMPLE, per_n=PER_N)

# Also keep an UNSTRATIFIED (natural corpus mix) real sample for the report,
# so the natural-mix mass distribution is visible too.
_, _, n_real_natural = sample_and_rasterise(DS0002, N_SAMPLE)

mass_real = occ_real.sum(axis=(1, 2))
mass_syn = occ_syn.sum(axis=(1, 2))

print(f"real   mass: {mass_real.mean():.2f} +/- {mass_real.std():.2f}  (n={len(mass_real)})")
print(f"synth  mass: {mass_syn.mean():.2f} +/- {mass_syn.std():.2f}  (n={len(mass_syn)})")

slices = descriptor_slices(8)
per_n_mass = {}
for n in (20, 50, 100):
    mr, ms = n_real == n, n_syn == n
    per_n_mass[n] = {
        "real_mean": float(mass_real[mr].mean()), "real_std": float(mass_real[mr].std()),
        "synthetic_mean": float(mass_syn[ms].mean()), "synthetic_std": float(mass_syn[ms].std()),
    }
    print(f"  n_objects={n}: real {mass_real[mr].mean():.1f}+/-{mass_real[mr].std():.1f}  "
          f"synthetic {mass_syn[ms].mean():.1f}+/-{mass_syn[ms].std():.1f}")

report = {
    "note": "sampled with a MATCHED n_objects mix (1/3 each of 20/50/100) in both "
            "real and synthetic, so pixel-mass/com scale comparably -- an unmatched "
            "n_objects mix alone would look like an out-of-distribution shift.",
    "n_sample_real": int(len(mass_real)),
    "n_sample_synthetic": int(len(mass_syn)),
    "pixel_mass": {
        "real_mean": float(mass_real.mean()), "real_std": float(mass_real.std()),
        "synthetic_mean": float(mass_syn.mean()), "synthetic_std": float(mass_syn.std()),
    },
    "pixel_mass_per_n_objects": per_n_mass,
    "descriptor_blocks": {},
}

flags = []
for name, sl in slices.items():
    if name == "_total":
        continue
    r = phi_real[:, sl]
    s = phi_syn[:, sl]
    r_lo, r_hi = r.min(axis=0), r.max(axis=0)
    s_lo, s_hi = s.min(axis=0), s.max(axis=0)
    # per-dim: does synthetic's [min,max] cover real's [min,max] (with 5% pad)?
    pad = 0.05 * np.maximum(r_hi - r_lo, 1e-8)
    covers = (s_lo <= r_lo + pad) & (s_hi >= r_hi - pad)
    frac_covered = float(covers.mean())
    report["descriptor_blocks"][name] = {
        "dim": int(sl.stop - sl.start),
        "real_range": [r_lo.tolist(), r_hi.tolist()],
        "synthetic_range": [s_lo.tolist(), s_hi.tolist()],
        "frac_dims_covered": frac_covered,
    }
    if frac_covered < 0.5:
        flags.append(name)

report["out_of_distribution_blocks"] = flags
print("descriptor block coverage (frac of dims where synthetic covers real range):")
for name, v in report["descriptor_blocks"].items():
    print(f"  {name:10s} dim={v['dim']:2d}  frac_covered={v['frac_dims_covered']:.2f}")
print("OUT-OF-DISTRIBUTION blocks (frac_covered < 0.5):", flags if flags else "none")

with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
    json.dump(report, fh, indent=2)

# ---------------------------------------------------------------------------
# Figure: pixel mass overlay + COM scatter + mass by n_objects
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
bins = np.linspace(min(mass_real.min(), mass_syn.min()), max(mass_real.max(), mass_syn.max()), 40)
ax.hist(mass_real, bins=bins, alpha=0.55, label=f"real (n={len(mass_real)})", color="#3b6fa0", density=True)
ax.hist(mass_syn, bins=bins, alpha=0.55, label=f"synthetic (n={len(mass_syn)})", color="#d97a3f", density=True)
ax.set_xlabel("pixel mass (occupied pixels / 64x64)")
ax.set_ylabel("density")
ax.set_title("Total occupied-pixel mass")
ax.legend(fontsize=8)

ax = axes[1]
com_sl = slices["com"]
ax.scatter(phi_real[:, com_sl][:, 1], phi_real[:, com_sl][:, 0], s=2, alpha=0.15, color="#3b6fa0", label="real")
ax.scatter(phi_syn[:, com_sl][:, 1], phi_syn[:, com_sl][:, 0], s=2, alpha=0.15, color="#d97a3f", label="synthetic")
ax.set_xlabel("com_x (grid frac)")
ax.set_ylabel("com_y (grid frac)")
ax.set_title("Centre of mass")
ax.legend(fontsize=8, markerscale=4)

ax = axes[2]
for n in sorted(set(n_real.tolist())):
    m = n_real == n
    ax.scatter(np.full(m.sum(), n) - 1.5, mass_real[m], s=3, alpha=0.15, color="#3b6fa0")
for n in sorted(set(n_syn.tolist())):
    m = n_syn == n
    ax.scatter(np.full(m.sum(), n) + 1.5, mass_syn[m], s=3, alpha=0.15, color="#d97a3f")
ax.set_xlabel("n_objects (real: -1.5, synthetic: +1.5 jitter)")
ax.set_ylabel("pixel mass")
ax.set_title("Mass by n_objects")

fig.suptitle("DS-0002 (real) vs DS-0003 (synthetic): occupancy descriptor comparison")
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "real_vs_synthetic.png")
fig.savefig(fig_path, dpi=130)
print(f"saved figure: {fig_path}")
