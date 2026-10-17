"""Summarise regret_decomp.py output by push stage. Value change units (lyapunov - in-goal fraction; lower = better). Per stage k:
executed = true value change of the push the closed-loop planner actually executed at that state (recorded episode; CEM-refined, legalised at execution),
best bank / best all = true best of the simulated candidate set C restricted to bank rows / all rows (incl. fetch), pick bank / pick all = true value of the
model's argmax over those rows, model regret = pick - best (same rows), proposal gain = best all - best bank, Spearman(pred, true) over C.
Means with bootstrap 95 % CI over states.
usage: python -u analyse_regret.py results/regret_base1F.json --src results/fix_H1full.json --cell base1F"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse import value_curve


def ci(x, rng):
    x = np.asarray(x, float); bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--src", required=True); ap.add_argument("--cell", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    S = json.loads((HERE / a.path).read_text())["states"] if not Path(a.path).is_absolute() else json.loads(Path(a.path).read_text())["states"]
    src = Path(a.src); src = src if src.is_absolute() else HERE / src
    eps = {(e["goal"], e["start"]): e for e in json.loads(src.read_text())["episodes"] if e["cell"] == a.cell}
    vc = {}
    for s in S:
        e = eps[(s["goal"], s["start"])]; key = (s["goal"], s["start"])
        if key not in vc:
            vc[key] = value_curve(e["states"], s["goal"])
        s["executed"] = float(vc[key][s["k"] + 1] - vc[key][s["k"]])
    rows = ["| push k | n | strays | executed (planner) | best bank | best all | pick bank | pick all | model regret (all) | proposal gain (best all - best bank) | pick all - executed | model pick is fetch | Spearman |", "|" + "---|" * 13]
    for k in sorted({s["k"] for s in S}) + ["all"]:
        T = [s for s in S if k == "all" or s["k"] == k]
        f = lambda key: [s[key] for s in T]
        rows.append(f"| {k} | {len(T)} | {np.mean(f('n_strays')):.1f} | {ci(f('executed'), rng)} | {ci(f('best_true_bank'), rng)} | {ci(f('best_true_all'), rng)} | {ci(f('pick_bank_true'), rng)} | {ci(f('pick_all_true'), rng)} | "
                    f"{ci(np.subtract(f('pick_all_true'), f('best_true_all')), rng)} | {ci(np.subtract(f('best_true_all'), f('best_true_bank')), rng)} | {ci(np.subtract(f('pick_all_true'), f('executed')), rng)} | "
                    f"{np.mean([s['pick_all_src'].startswith('fetch') for s in T]):.2f} | {np.nanmean(f('spearman')):+.2f} |")
    md = [f"# Regret decomposition ({a.path}, states from {a.cell}; value change, lower = better; 64 simulated candidates per state)\n"] + rows
    out = HERE / "results" / (Path(a.path).stem + "_summary.md"); out.write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
