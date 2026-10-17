"""EXP-0067: placement optimum (EXP-0046 vstar.py, same solver and scoring) for the widened letter goals
letter_{O,T,S,X}_w{10,15,20}. Writes results/vstar_width.json (checkpointed per goal); also re-derives the 4 baseline
letters as an identity check against EXP-0046 results/vstar.json. Usage: python -u code/vstar_width.py"""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import distance_transform_edt
REPO = Path(__file__).resolve().parents[3]; os.chdir(REPO); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0046-goal-ceiling/code"))
import vstar as V  # noqa: E402  (EXP-0046 solver; its main() is not called)
from Baselines.common.goals import dist_field_from_mask, letter_goal_mask  # noqa: E402
from simple_mpc.adapters import OCC_FOOTPRINT_RADIUS, OCC_GRID  # noqa: E402
OUT = REPO / "experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json"
GOALS = [f"letter_{L}" for L in "OTSX"] + [f"letter_{L}_w{w}" for w in (10, 15, 20) for L in "OTSX"]


def solve_goal(mask):
    """EXP-0046 solver + scoring on one (64, 64) goal mask -> the vstar_width.json record (factored out 2026-10-15 so EXP-0075 vstar_disc.py reuses it)."""
    total_mass = V.N * math.pi * OCC_FOOTPRINT_RADIUS ** 2
    mask_t = torch.from_numpy(mask.astype(np.float32)); dist_t = torch.from_numpy(dist_field_from_mask(mask > 0)).float()
    D = dist_t.numpy().astype(np.float64); Cl = V.per_cube_map(D); Cm = -V.per_cube_map(mask.astype(np.float64))
    p_l = V.optimise(Cl); p_m = min([V.optimise(Cm), V.refine(Cm, p_l)], key=lambda p: sum(Cm[q] for q in p))
    inside = V.fine_inside(mask); edt = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    r = dict(vstar=V.evaluate(p_l, mask_t, dist_t), best_mass_placement=V.evaluate(p_m, mask_t, dist_t),
             nonoverlap_ok=V.check_nonoverlap(p_l) and V.check_nonoverlap(p_m), n_inside_vstar=int(sum(inside[q] for q in p_l)),
             capacity=V.capacity(inside), mask_px=int(mask.sum()), stroke_width_max_mm=float(2 * edt.max() * V.PITCH_SCORE * 1e3),
             stroke_width_median_skeleton_mm=float(2 * np.median(edt[sk]) * V.PITCH_SCORE * 1e3), theoretical_max_mass=total_mass,
             best_mass_centres_m=[[float(V.XS[i]), float(V.XS[j])] for i, j in p_m])
    r["mass_frac_at_vstar"] = r["vstar"]["mass_in_region"] / total_mass
    r["mass_frac_best_placement"] = r["best_mass_placement"]["mass_in_region"] / total_mass
    return r


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    total_mass = V.N * math.pi * OCC_FOOTPRINT_RADIUS ** 2
    for g in GOALS:
        if g in res:
            continue
        r = solve_goal(letter_goal_mask(g, OCC_GRID, OCC_GRID))
        res[g] = r; tmp = str(OUT) + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
        print(f"{g:14s} cap={r['capacity']:3d} in={r['n_inside_vstar']:2d} w_med={r['stroke_width_median_skeleton_mm']:.1f} "
              f"w_max={r['stroke_width_max_mm']:.1f} px={r['mask_px']} best_massfrac={r['mass_frac_best_placement']:.4f}", flush=True)
    base = json.load(open("experiments/EXP-0046-goal-ceiling/results/vstar.json"))
    for L in "OTSX":
        print(f"identity check letter_{L}: here {res[f'letter_{L}']['mass_frac_best_placement']:.5f} vs EXP-0046 {base[f'letter_{L}']['mass_frac_best_placement']:.5f}")


if __name__ == "__main__":
    main()
