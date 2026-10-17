"""EXP-0075 T0b (2026-10-15): placement optimum V* for the gather-to-disc goals (Baselines/common/goals.py::named_goal_mask 'disc_*'), with the EXP-0046
solver through EXP-0067 vstar_width.solve_goal (same solver, scoring and record fields), so analyse.rel_curve works on disc goals (rel = in-goal mass /
mass_frac_best_placement). Identity check: re-solves letter_O_w20 and compares with vstar_width.json. Writes results/vstar_disc.json (atomic, per goal).
usage (repo root): python -u experiments/EXP-0075-closed-loop-benchmark/code/vstar_disc.py [disc_n20 ...]"""
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; os.chdir(REPO); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0067-letter-stroke-width-sweep/code"))
from vstar_width import solve_goal  # noqa: E402  (also puts the EXP-0046 solver on the path)
from Baselines.common.goals import named_goal_mask, parse_disc_goal  # noqa: E402
from simple_mpc.adapters import OCC_GRID  # noqa: E402
OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/vstar_disc.json"


def main():
    goals = sys.argv[1:] or ["disc_n20"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ref = json.loads((REPO / "experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json").read_text())["letter_O_w20"]["mass_frac_best_placement"]
    chk = solve_goal(named_goal_mask("letter_O_w20", OCC_GRID, OCC_GRID))["mass_frac_best_placement"]
    print(f"identity check letter_O_w20: here {chk:.6f} vs vstar_width.json {ref:.6f}", flush=True); assert abs(chk - ref) < 1e-6
    for g in goals:
        if g in res:
            continue
        r = solve_goal(named_goal_mask(g, OCC_GRID, OCC_GRID)); r["radius_m"] = parse_disc_goal(g); r["centre_m"] = [0.0, 0.0]
        res[g] = r; tmp = str(OUT) + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
        print(f"{g}: r={1000 * r['radius_m']:.2f} mm px={r['mask_px']} cap={r['capacity']} in={r['n_inside_vstar']} best_massfrac={r['mass_frac_best_placement']:.4f}", flush=True)


if __name__ == "__main__":
    main()
