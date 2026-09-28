"""EXP-0057 analysis. Reads every results/oracle_*.json cell produced by code/queue.py
(and oracle.py directly) and computes, per cell:
  - completion push k* (METRICS.md-style): first push with in-goal mass fraction >=
    0.9 x EXP-0046 vstar `mass_frac_best_placement`, else censored at 20 (--steps).
  - fraction solved, solved-only median pushes, and a CENSORED MEAN (unsolved -> 21,
    i.e. one more than the 20-push cap) -- see METRICS.md `oracle_completion_pushes`.
  - per-goal table, and paired differences vs the default cell (oracle_A_default) per
    (goal, start) task with bootstrap 95% CI (percentile, 4000 resamples).
  - the in-goal-mass curve vs push (mean and per-goal).
Also loads EXP-0055's `lyap` cell (same goals, starts 40/41, worldframe NFD GD 1s) as a
learned-model reference row, read-only (not re-analysed).
Writes results/analysis.json; prints a summary table.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from simple_mpc.adapters import occ_for_scoring
import torch

HERE = Path(__file__).resolve().parents[1]
VSTAR = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
CENSOR_STEPS = 20
CENSOR_VALUE = CENSOR_STEPS + 1  # METRICS.md oracle_completion_pushes: unsolved -> steps+1
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def completion_k(mass_frac, goal):
    opt = VSTAR[goal]["mass_frac_best_placement"]
    thresh = 0.9 * opt
    hit = [k for k, m in enumerate(mass_frac) if m >= thresh]
    return (hit[0], True) if hit else (CENSOR_VALUE, False)


def load_cell(path):
    d = json.loads(path.read_text())
    rows = []
    for e in d["episodes"]:
        k, solved = completion_k(e["mass_frac"], e["goal"])
        rows.append(dict(cell=e.get("cell", path.stem), goal=e["goal"], start=e["start"],
                         mass_frac=e["mass_frac"], values=e["values"],
                         signed_frac=e.get("signed_frac"), k=k, solved=solved,
                         sims_used=e.get("sims_used"), plan_time_s=e.get("plan_time_s")))
    return rows, d.get("step", 0)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "results")
    cells = sorted(results_dir.glob("oracle_*.json"))
    all_rows = {}
    for p in cells:
        rows, step = load_cell(p)
        if not rows:
            continue
        tag = rows[0]["cell"]
        all_rows[tag] = dict(rows=rows, step=step)
        print(f"{tag}: {len(rows)} episodes, {step} pushes completed")

    out = {"cells": {}, "paired_vs_default": {}, "per_goal": {}, "learned_reference": None}
    for tag, d in all_rows.items():
        R = d["rows"]
        ks = [r["k"] for r in R]; solved = [r["solved"] for r in R]
        solved_ks = [k for k, s in zip(ks, solved) if s]
        out["cells"][tag] = dict(
            n=len(R), steps_completed=d["step"], frac_solved=float(np.mean(solved)),
            median_pushes_solved_only=float(np.median(solved_ks)) if solved_ks else None,
            censored_mean_pushes=float(np.mean(ks)),
            mean_mass_frac_final=float(np.mean([r["mass_frac"][-1] for r in R])),
            mean_lyap_final=float(np.mean([r["values"][-1] for r in R])))

    default = all_rows.get("oracle_A_default")
    if default:
        base = {(r["goal"], r["start"]): r for r in default["rows"]}
        for tag, d in all_rows.items():
            if tag == "oracle_A_default":
                continue
            pairs = [(r, base[(r["goal"], r["start"])]) for r in d["rows"]
                    if (r["goal"], r["start"]) in base]
            if not pairs:
                continue
            dk = [a["k"] - b["k"] for a, b in pairs]
            dm = [a["mass_frac"][-1] - b["mass_frac"][-1] for a, b in pairs]
            out["paired_vs_default"][tag] = dict(
                n=len(pairs), delta_completion_pushes=dict(mean=float(np.mean(dk)), ci=ci(dk)),
                delta_mass_frac_final=dict(mean=float(np.mean(dm)), ci=ci(dm)))

    for tag, d in all_rows.items():
        for r in d["rows"]:
            out["per_goal"].setdefault(r["goal"], {}).setdefault(tag, []).append(
                dict(k=r["k"], solved=r["solved"], mass_frac_final=r["mass_frac"][-1]))
    for g, cellmap in out["per_goal"].items():
        for tag, rs in cellmap.items():
            out["per_goal"][g][tag] = dict(
                frac_solved=float(np.mean([x["solved"] for x in rs])),
                mean_mass_frac_final=float(np.mean([x["mass_frac_final"] for x in rs])),
                mean_k=float(np.mean([x["k"] for x in rs])))

    ref_p = REPO / "experiments/EXP-0055-capacity-aware-value/results/analysis_main.json"
    if ref_p.exists():
        ref = json.loads(ref_p.read_text())
        out["learned_reference"] = ref.get("summary", {}).get("lyap")

    p = results_dir / "analysis.json"
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(p)

    print(f"\n{'cell':30s} {'n':>3s} {'steps':>6s} {'frac_solved':>11s} {'med_k(solved)':>13s} "
          f"{'censored_mean_k':>15s} {'mass_frac_final':>15s}")
    for tag, c in out["cells"].items():
        print(f"{tag:30s} {c['n']:3d} {c['steps_completed']:6d} {c['frac_solved']:11.2f} "
              f"{str(c['median_pushes_solved_only']):>13s} {c['censored_mean_pushes']:15.2f} "
              f"{c['mean_mass_frac_final']:15.3f}")
    print("\npaired vs oracle_A_default:")
    for tag, d in out["paired_vs_default"].items():
        dc, dm = d["delta_completion_pushes"], d["delta_mass_frac_final"]
        print(f"  {tag:30s} n={d['n']:2d}  delta_k {dc['mean']:+.2f} [{dc['ci'][0]:+.2f},{dc['ci'][1]:+.2f}]"
              f"  delta_mass {dm['mean']:+.3f} [{dm['ci'][0]:+.3f},{dm['ci'][1]:+.3f}]")
    if out["learned_reference"]:
        print("\nlearned-model reference (EXP-0055 lyap cell, k20):", out["learned_reference"].get("k20"))


if __name__ == "__main__":
    main()
