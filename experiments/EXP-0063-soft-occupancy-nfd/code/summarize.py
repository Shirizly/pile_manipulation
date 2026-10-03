"""EXP-0063: one table from results/pilot_eval.json (1-step, DS-0016 test + DS-0017 val) and
results/multistep.json + multistep_raw.json (DS-0018 3-push terminal), with paired pool-
bootstrap deltas vs the per-pool mean of the 3 baseline seeds. Writes results/summary.md."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_pilot import BASE, paired, _cap_matrix, comparisons  # noqa: E402

R = HERE.parent / "results"
DEGEN_FRAC = 0.05  # sensitivity: drop (pool, goal) cells whose TRUE dv spread (mean - min) is
#                    < 5% of the split's median spread -- capture divides by that spread, so a
#                    near-tie pool turns one bad pick into a capture of -40 (val pool 22,
#                    quadrant_0). vt is model-independent: the same cells drop for every model.


def drop_degenerate(raw_goal_dict):
    dens = [np.mean(r["vt"]) - np.min(r["vt"]) for rows in raw_goal_dict.values() for r in rows]
    thr = DEGEN_FRAC * float(np.median(dens))
    return {g: [r for r in rows if np.mean(r["vt"]) - np.min(r["vt"]) >= thr]
            for g, rows in raw_goal_dict.items()}, thr


def robust_pooled(ev):
    """Pooled 64-pool comparisons with near-degenerate cells dropped (per split threshold)."""
    pooled, n_drop = {}, 0
    for m, e in ev.items():
        if m.startswith("_") or not all(s in e for s in ("test", "val")):
            continue
        raw = {}
        for s, off in (("test", 0), ("val", 10_000)):
            kept, _ = drop_degenerate(e[s]["raw"]["slateN_tough_raw"])
            n_drop = sum(len(v) for v in e[s]["raw"]["slateN_tough_raw"].values()) - sum(len(v) for v in kept.values())
            for g, rows in kept.items():
                raw.setdefault(g, []).extend({**r, "pool": r["pool"] + off} for r in rows)
        pooled[m] = {"pooled": {"raw": {"slateN_tough_raw": raw}}}
    return comparisons(pooled, "pooled")


def main():
    ev = json.loads((R / "pilot_eval.json").read_text())
    ms = json.loads((R / "multistep.json").read_text()) if (R / "multistep.json").exists() else {}
    msr = json.loads((R / "multistep_raw.json").read_text()) if (R / "multistep_raw.json").exists() else {}

    ms_caps = {m: _cap_matrix(v["slateN_tough_terminal_raw"])[0] for m, v in msr.items()}
    ms_base = (np.nanmean(np.stack([ms_caps[b] for b in BASE]), 0)
               if all(b in ms_caps for b in BASE) else None)

    rob = robust_pooled(ev)
    rob_val = {}
    for m, e in ev.items():
        if not m.startswith("_") and "val" in e:
            kept, thr = drop_degenerate(e["val"]["raw"]["slateN_tough_raw"])
            caps = _cap_matrix(kept)[0]
            rob_val[m] = float(np.nanmean(np.nanmean(caps, 1)))

    def d(block, m, ref="vs_seed_mean"):
        x = (block or {}).get(m, {}).get(ref)
        return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" if x else "--"

    rows = ["| model | test slateN_tough | val slateN_tough | test slateN (13g) | 3-push term. slateN_tough | "
            "Δ vs 3-seed mean: test | val | pooled 64 | 3-push | val slateN_tough, degen. dropped | Δ pooled, degen. dropped | Δ vs seed 0, pooled 64 | acc1* | rollout4* | mean p(1-p) | frac mid px | mass ratio |",
            "|" + "---|" * 17]
    models = [m for m in ev if not m.startswith("_")]
    for m in models:
        t = ev[m].get("test", {}).get("metrics", {}); v = ev[m].get("val", {}).get("metrics", {})
        b = ev[m].get("binariness") or {}
        msd = "--"
        if ms_base is not None and m in ms_caps and m not in BASE and m != "persistence":
            x = paired(ms_base, ms_caps[m])
            msd = f"{x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
        f = lambda x: "--" if x is None else f"{x:.3f}"
        rows.append(
            f"| {m} | {f(t.get('slateN_tough'))} | {f(v.get('slateN_tough'))} | {f(t.get('slateN'))} | "
            f"{f(ms.get(m, {}).get('terminal_slateN_tough'))} | {d(ev.get('_paired_test'), m)} | "
            f"{d(ev.get('_paired_val'), m)} | {d(ev.get('_paired_pooled'), m)} | {msd} | "
            f"{f(rob_val.get(m))} | {d(rob, m)} | "
            f"{d(ev.get('_paired_pooled'), m, 'vs_nfd_3ch_narrow_l20_v2')} | "
            f"{f(t.get('accuracy_1'))} | {f(t.get('rollout_accuracy_4'))} | {f(b.get('mean_p1mp'))} | "
            f"{f(b.get('frac_mid'))} | {f(b.get('mass_ratio'))} |")
    txt = ("# EXP-0063 pilot summary\n\nslateN_tough = 8 tough goals, lyapunov, soft-splat truth. "
           "Δ = paired pool-bootstrap (2000 draws) vs the per-pool mean capture of the 3 baseline "
           "seeds; 95% CI.\n*acc1/rollout4 are scored against the HARD raster: not comparable for "
           "soft_* arms or outblur controls.\n'degen. dropped' = sensitivity check excluding (pool, goal) cells whose true "
           "dv spread is < 5% of the split median (3 val quadrant_0 cells, 0 test cells).\n\n" + "\n".join(rows) + "\n")
    (R / "summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
