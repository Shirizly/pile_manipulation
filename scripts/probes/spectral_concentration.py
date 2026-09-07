"""Spectral concentration of a lyapunov_weights field: what fraction of its
|FFT|^2 sits within radius r of DC (64x64 grid).

Motivation (EXP-0024_v2/EXP-0026_v2): every control-utility number in this
register is computed from ONE cost functional, V = d^T y / ||y||_1, with `d`
a distance transform. A distance transform is low-pass by construction (it is
built from a Euclidean-distance field, which is smooth everywhere except at
the target boundary), so this measures directly whether that is true, and by
how much, before attributing any control-utility result to "the model" versus
"the metric can't see fine detail."

Because |FFT(x - x0)| = |FFT(x)| (the shift theorem only changes phase), a
target's on-grid POSITION never affects this measurement -- only its shape/
size and the functional applied to it (distance transform vs indicator vs
clipped-distance) does. So `ind-square8`'s off-centre placement (flush in the
corner, for the C-040 degeneracy reason -- see control_utility_test.py) gives
the identical concentration numbers a centred 8x8 square would.

Usage
-----
    PYTHONPATH=. python scripts/probes/spectral_concentration.py
"""
from __future__ import annotations

import numpy as np

from control_utility_test import lyapunov_weights

RADII = (1, 4, 8)


def concentration(field: np.ndarray, radii=RADII):
    """Fraction of |FFT|^2 within radius r (grid units) of DC, for each r.

    The DC bin (mean value) is removed BEFORE the ratio is formed -- both
    from the numerator and the total. Every field here is a low-pass-shaped
    but non-zero-mean field (a distance transform is mostly large positive
    values; an indicator is mostly 0/1), so its raw, un-demeaned power is
    dominated by the mean almost independently of shape, which would make
    every field in this table read as ~concentrated at DC regardless of how
    much fine structure it has. Demeaning isolates the part of the spectrum
    that actually varies over the grid -- the part a convolutional model's
    receptive field or a metric's spatial resolution can or cannot see."""
    Fmean = field.astype(np.float64).mean()
    F = np.fft.fft2(field.astype(np.float64) - Fmean)
    P = np.abs(F) ** 2
    P[0, 0] = 0.0
    H, W = field.shape
    fy = np.fft.fftfreq(H) * H   # integer cycle counts, DC at 0
    fx = np.fft.fftfreq(W) * W
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    total = P.sum()
    return {r: float(P[R <= r].sum() / total) for r in radii}


def indicator_annulus(H, W, r_in, r_out):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((r >= r_in) & (r < r_out)).astype(np.float32)


def main():
    H = W = 64
    fields = {
        "distance transform, corner half-plane": lyapunov_weights((H, W), "corner", "cpu").numpy(),
        "distance transform, eighth-side square (dist-square8)": lyapunov_weights((H, W), "dist-square8", "cpu").numpy(),
        "indicator, corner half-plane (ind-corner)": lyapunov_weights((H, W), "ind-corner", "cpu").numpy(),
        "indicator, 8x8 px square (ind-square8)": lyapunov_weights((H, W), "ind-square8", "cpu").numpy(),
        "indicator, annulus r 10-16 px (context only, not a goal key)": indicator_annulus(H, W, 10, 16),
        "distclip-corner-r2": lyapunov_weights((H, W), "distclip-corner-r2", "cpu").numpy(),
        "distclip-corner-r4": lyapunov_weights((H, W), "distclip-corner-r4", "cpu").numpy(),
        "distclip-corner-r8": lyapunov_weights((H, W), "distclip-corner-r8", "cpu").numpy(),
    }
    print(f"{'weight field':58s} " + "  ".join(f"r<={r}" for r in RADII))
    for name, f in fields.items():
        c = concentration(f)
        row = "  ".join(f"{c[r]:.3f}" for r in RADII)
        print(f"{name:58s} {row}")


if __name__ == "__main__":
    main()
