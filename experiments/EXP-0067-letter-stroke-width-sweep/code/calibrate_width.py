"""EXP-0067: realised stroke width on the 64 px scoring grid vs asset dilation d (720 px glyph), letters O/T/S/X.
Measure = EXP-0046 vstar.py's: 2 x median EDT on the skeleton x PITCH_SCORE (mm); also max width and mask px.
Picks d per nominal width (10/15/20 mm) = argmin |mean_letters(median width) - target|. Writes results/width_calibration.json.
Usage: python -u code/calibrate_width.py"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import widened_letter_mask  # noqa: E402
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID  # noqa: E402
PITCH = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / (OCC_GRID - 1)
LETTERS = ["O", "T", "S", "X"]; TARGETS = [10, 15, 20]; OUT = REPO / "experiments/EXP-0067-letter-stroke-width-sweep/results/width_calibration.json"


def widths(mask):
    edt = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]; sk = skeletonize(mask)
    return dict(median_mm=float(2 * np.median(edt[sk]) * PITCH * 1e3), max_mm=float(2 * edt.max() * PITCH * 1e3), px=int(mask.sum()))


def main():
    table = {}
    for d in range(0, 46):
        table[d] = {L: widths(widened_letter_mask(L, OCC_GRID, OCC_GRID, d)) for L in LETTERS}
        table[d]["mean_median_mm"] = float(np.mean([table[d][L]["median_mm"] for L in LETTERS]))
        print(d, f"mean {table[d]['mean_median_mm']:.2f}", " ".join(f"{L}:{table[d][L]['median_mm']:.2f}/{table[d][L]['max_mm']:.1f}/{table[d][L]['px']}" for L in LETTERS), flush=True)
    pick = {t: int(min(table, key=lambda d: abs(table[d]["mean_median_mm"] - t))) for t in TARGETS}
    print("picked d:", pick, {t: table[pick[t]] for t in TARGETS})
    OUT.write_text(json.dumps(dict(pitch_mm=PITCH * 1e3, table=table, picked=pick, realised={str(t): table[pick[t]] for t in TARGETS},
                                   baseline=table[0]), indent=1))


if __name__ == "__main__":
    main()
