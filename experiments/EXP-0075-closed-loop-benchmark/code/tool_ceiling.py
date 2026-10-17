"""P0-7b tool-conditioned ceiling (RAL plan, 2026-10-15; CPU only, no Genesis).

Question: does the 40 mm blade itself cap how much of a letter goal can be filled? Model: a cube can only be left where the
blade's contact footprint at the push-END pose fits inside the goal. Footprint = the blade rectangle (2*BLADE_HL = 40 mm long,
2*BLADE_HW = 2 mm thick; wide_planner constants), at EVERY heading (the 4-D push action space allows any push direction, the
blade is perpendicular to it; 180-deg periodic -> 0..178 deg in 2-deg steps). Tool-reachable region = union over headings of
the morphological OPENING of the goal mask by that rotated rectangle (pixels covered by at least one placement of the footprint
lying wholly inside the goal). The rectangle is rasterised on the 64x64 scoring grid (pixel-centre pitch PITCH_SCORE = 2.03 mm):
offset pixel (di, dj) is in the element when its centre lies within |along| <= HL, |across| <= max(HW, half a pixel) -- 2 mm
is below one pixel, so the element is a 1-px digital line.
Sensitivity variant `band7`: footprint = blade + the row of cubes it pushes (40 mm x (2 + 5) mm), the stricter reading
"the blade face and the cube it deposits must both sit in the goal".

V*_tool: the EXP-0046 placement solver (through EXP-0067 vstar_width.solve_goal's pieces) maximising in-goal mass of 20 cubes
against the OPENED mask, then scored against the ORIGINAL mask (in-goal mass fraction, the quantity rel divides).
rel_ceiling = V*_tool mass fraction / V* mass fraction (vstar_width.json). Also: capacity = max number of 5 mm cube footprints
wholly inside the opened mask (EXP-0046 capacity()).
Writes results/tool_ceiling.json (checkpointed per goal x variant). Usage: CUDA_VISIBLE_DEVICES= python -u code/tool_ceiling.py"""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_dilation, binary_erosion
REPO = Path(__file__).resolve().parents[3]; os.chdir(REPO); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0046-goal-ceiling/code"))
import vstar as V  # noqa: E402  (EXP-0046 solver; main() not called)
from Baselines.common.goals import dist_field_from_mask, letter_goal_mask  # noqa: E402
from simple_mpc.adapters import OCC_FOOTPRINT_RADIUS, OCC_GRID  # noqa: E402

OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/tool_ceiling.json"
VSTAR = json.loads((REPO / "experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json").read_text())
GOALS = [f"letter_{L}_w{w}" for w in (20, 15, 10) for L in "OTSX"]
BLADE_HL, BLADE_HW, CUBE = 0.02, 0.001, 0.005          # == wide_planner.BLADE_HL, BLADE_HW; cube side
VARIANTS = {"blade": (BLADE_HL, BLADE_HW), "band7": (BLADE_HL, BLADE_HW + CUBE / 2)}   # band7: half-thickness 3.5 mm (2 + 5 mm)
HEADINGS = np.deg2rad(np.arange(0, 180, 2))


def footprint(theta, hl, hw, pitch=V.PITCH_SCORE):
    r = int(math.ceil((hl + hw) / pitch)) + 1
    di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    x, y = di * pitch, dj * pitch
    along = x * math.cos(theta) + y * math.sin(theta); across = -x * math.sin(theta) + y * math.cos(theta)
    return (np.abs(along) <= hl + 1e-9) & (np.abs(across) <= max(hw, pitch / 2) + 1e-9)


def tool_open(mask, hl, hw):
    """union over headings of binary_opening(mask, rotated footprint); border = outside the goal."""
    out = np.zeros_like(mask, dtype=bool)
    for th in HEADINGS:
        se = footprint(th, hl, hw)
        er = binary_erosion(mask, structure=se, border_value=0)
        out |= binary_dilation(er, structure=se)
    return out & mask


def solve(mask, opened):
    total = V.N * math.pi * OCC_FOOTPRINT_RADIUS ** 2
    mask_t = torch.from_numpy(mask.astype(np.float32)); dist_t = torch.from_numpy(dist_field_from_mask(mask > 0)).float()
    if not opened.any():
        return dict(mass_frac_tool=0.0, capacity_tool=0, n_inside_tool=0)
    Cm = -V.per_cube_map(opened.astype(np.float64))
    p = V.optimise(Cm); p = min([p, V.refine(Cm, p)], key=lambda q: sum(Cm[x] for x in q))
    ins = V.fine_inside(opened)
    ev = V.evaluate(p, mask_t, dist_t)                          # scored against the ORIGINAL goal
    ev_open = V.evaluate(p, torch.from_numpy(opened.astype(np.float32)), dist_t)
    return dict(mass_frac_tool=ev["mass_in_region"] / total, mass_frac_in_opened=ev_open["mass_in_region"] / total,
                capacity_tool=int(V.capacity(ins)), n_inside_tool=int(sum(ins[q] for q in p)), nonoverlap_ok=bool(V.check_nonoverlap(p)),
                centres_m=[[float(V.XS[i]), float(V.XS[j])] for i, j in p])


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("_meta", dict(headings_deg=[0, 178, 2], pitch_mm=V.PITCH_SCORE * 1e3, variants={k: [2e3 * a, 2e3 * b] for k, (a, b) in VARIANTS.items()},
                                 note="variants: footprint length x thickness in mm; rel_ceiling = mass_frac_tool / vstar_width mass_frac_best_placement"))
    for g in GOALS:
        mask = letter_goal_mask(g, OCC_GRID, OCC_GRID).astype(bool)
        for vn, (hl, hw) in VARIANTS.items():
            key = f"{g}|{vn}"
            if key in res:
                continue
            op = tool_open(mask, hl, hw)
            r = dict(goal=g, variant=vn, mask_px=int(mask.sum()), opened_px=int(op.sum()), removed_px=int(mask.sum() - op.sum()),
                     removed_px_idx=np.argwhere(mask & ~op).tolist(), vstar_mass_frac=VSTAR[g]["mass_frac_best_placement"],
                     vstar_capacity=VSTAR[g]["capacity"])
            r.update(solve(mask, op))
            r["rel_ceiling"] = r["mass_frac_tool"] / r["vstar_mass_frac"]
            res[key] = r
            tmp = str(OUT) + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
            print(f"{g:14s} {vn:6s} px {r['mask_px']:4d} -> {r['opened_px']:4d} (-{r['removed_px']:3d})  cap {r['vstar_capacity']:3d} -> {r['capacity_tool']:3d}  "
                  f"mass {r['vstar_mass_frac']:.4f} -> {r['mass_frac_tool']:.4f}  rel_ceiling {r['rel_ceiling']:.4f}", flush=True)


if __name__ == "__main__":
    main()
