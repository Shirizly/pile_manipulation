"""EXP-0049 analysis (CPU): artifacts/sim/*.pt -> results/sampler_study.json."""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import dist_field_from_mask, letter_mask, quadrant_mask, higher_is_better_for
from simple_mpc.adapters import occ_for_scoring, OCC_GRID

EXP = REPO / "experiments/EXP-0049-sampler-study"
SIM = EXP / "artifacts/sim"
GOALS = [f"letter_{c}" for c in "OTSLCXHZI"] + [f"quadrant_{q}" for q in (0, 1, 3)]
SAMPLERS = ["S1_pile", "S1_pile_fulllen", "S2_pile_relaxed", "S2_pile_relaxed_fulllen", "S3_placement_perp", "S4_blind_perp"]
NS = [8, 16, 32, 64, 96]
MIX = {"MIX_S3_75_S1_25": {"S3_placement_perp": 0.75, "S1_pile": 0.25},
       "MIX_S3_50_S1_50": {"S3_placement_perp": 0.5, "S1_pile": 0.5},
       "MIX_S3_75_S1full_25": {"S3_placement_perp": 0.75, "S1_pile_fulllen": 0.25},
       "MIX_S3_50_S1full_50": {"S3_placement_perp": 0.5, "S1_pile_fulllen": 0.5}}
R = 400


def goal_tensors():
    masks, dists = [], []
    for g in GOALS:
        m = quadrant_mask(OCC_GRID, OCC_GRID, int(g[-1])) if g.startswith("quadrant_") \
            else letter_mask(g[-1], OCC_GRID, OCC_GRID)
        masks.append(torch.from_numpy(m.astype(np.float32)))
        dists.append(torch.from_numpy(dist_field_from_mask(m)).float())
    return torch.stack(masks), torch.stack(dists)


def lyap(occ, dists):
    f = occ.reshape(occ.shape[0], -1)
    return (f @ dists.reshape(len(dists), -1).T) / f.sum(1, keepdim=True).clamp_min(1e-6)


def best_of_n(imp, n, rng, weights_pools=None):
    """imp (C,G) -> mean over R random n-subsets of max improvement, (G,)."""
    C = imp.shape[0]
    if n >= C:
        return imp.max(0)
    idx = np.stack([rng.choice(C, n, replace=False) for _ in range(R)])
    return imp[idx].max(1).mean(0)


def main():
    assert not higher_is_better_for("lyapunov")
    _, dists = goal_tensors()
    rng = np.random.default_rng(0)
    rows, raw_imp = {}, {}
    for p in sorted(SIM.glob("*.pt")):
        name, sname = p.stem.split("_", 1)
        d = torch.load(p, weights_only=False)
        rl0, pe0 = d["raw_length"], d["raw_perp_err"]
        st_all, fin_all, fl_all = d["state"].float(), d["final"].float(), d["final_len"]
        variants = [(sname, torch.ones(len(rl0), dtype=torch.bool))]
        if sname.startswith("S1") or sname.startswith("S2"):
            # emulate a redraw-on-short sampler: keep only pushes the sampler itself made 20 mm
            variants.append((sname + "_fulllen", (rl0 - 0.02).abs() < 1e-4))
        for lab, keep in variants:
            st, fin, rl, pe = st_all, fin_all[keep], rl0[keep], pe0[keep]
            d = dict(d, final_len=fl_all[keep])
            disp = (fin[:, :, :2] - st[None, :, :2]).norm(dim=-1)            # (C,P)
            moved = (disp > 1e-3).sum(1).float()
            occ0 = occ_for_scoring(st[None], device="cpu")
            occ1 = occ_for_scoring(fin, device="cpu")
            imp = (lyap(occ0, dists) - lyap(occ1, dists)).numpy()          # (C,G) >0 = better
            raw_imp[(name, lab)] = imp
            o = occ1.reshape(len(occ1), -1)
            o = o / o.sum(1, keepdim=True)
            pair = torch.cdist(o, o, p=1)
            C = len(o)
            live = rl > 1e-3
            rows[(name, lab)] = {
                "state": name, "type": "scatter" if name.startswith("scatter") else "clump",
                "sampler": lab, "n_cand": int(len(fin)),
                "null_frac": float((moved == 0).float().mean()),
                "ge3_frac": float((moved >= 3).float().mean()),
                "cubes_moved_mean": float(moved.mean()),
                "disp_sum_mm_mean": float(disp.sum(1).mean() * 1e3),
                "disp_max_mm_mean": float(disp.max(1).values.mean() * 1e3),
                "cube_coverage": float((disp > 1e-3).any(0).float().mean()),
                "outcome_L1_pairwise_mean": float(pair.sum() / (C * (C - 1))),
                "imp_mean": float(imp.mean()),
                "imp_best96": float(imp.max(0).mean()),
                "imp_std_over_cands": float(imp.std(0).mean()),
                "frac_improving": float((imp > 0).mean()),
                **{f"best_of_{n}": float(best_of_n(imp, n, rng).mean()) for n in NS},
                "viol_raw_len_off_gt0.1mm": int(((rl - 0.02).abs() > 1e-4).sum()),
                "viol_raw_len_zero": int((~live).sum()),
                "viol_raw_perp_gt1e-3rad_live": int((pe[live] > 1e-3).sum()),
                "raw_perp_err_max_live": float(pe[live].max()) if live.any() else None,
                "raw_len_min_mm": float(rl.min() * 1e3),
                "final_len_off_gt1e-6": int(((d["final_len"] - 0.02).abs() > 1e-6).sum()),
                "t_sample_s": d["t_sample"], "t_sim_s": d["t_sim"],
            }
    # keep only states with all four samplers simulated (run was cut by the time budget)
    base = ["S1_pile", "S2_pile_relaxed", "S3_placement_perp", "S4_blind_perp"]
    complete = {k[0] for k in rows if all((k[0], b) in rows for b in base)}
    rows = {k: v for k, v in rows.items() if k[0] in complete}
    # mixtures (resampled from the pools actually simulated)
    states = sorted(complete)
    for mname, w in MIX.items():
        for s in states:
            if not all((s, k) in raw_imp for k in w):
                continue
            out = {}
            for n in NS[:-1]:
                vals = []
                for _ in range(R):
                    parts = [raw_imp[(s, k)][rng.choice(len(raw_imp[(s, k)]), int(round(f * n)), replace=False)]
                             for k, f in w.items()]
                    vals.append(np.concatenate(parts).max(0))
                out[f"best_of_{n}"] = float(np.mean(vals, 0).mean())
            rows[(s, mname)] = {"state": s, "type": "scatter" if s.startswith("scatter") else "clump",
                                "sampler": mname, **out}
    # aggregate + paired vs S3
    agg = {}
    keys = ["null_frac", "ge3_frac", "cubes_moved_mean", "disp_sum_mm_mean", "cube_coverage",
            "outcome_L1_pairwise_mean", "imp_mean", "imp_best96", "frac_improving"] + \
        [f"best_of_{n}" for n in NS] + ["t_sample_s", "t_sim_s"]
    for typ in ("scatter", "clump"):
        for sname in SAMPLERS + list(MIX):
            rs = [r for r in rows.values() if r["type"] == typ and r["sampler"] == sname]
            if not rs:
                continue
            a = {"n_states": len(rs)}
            for k in keys:
                v = [r[k] for r in rs if k in r]
                if v:
                    a[k] = float(np.mean(v))
            for k in ["viol_raw_len_off_gt0.1mm", "viol_raw_len_zero", "viol_raw_perp_gt1e-3rad_live",
                      "final_len_off_gt1e-6"]:
                v = [r[k] for r in rs if k in r]
                if v:
                    a[k] = int(np.sum(v))
            # paired difference vs S3 on best_of_16 / best96
            for k in ["best_of_16", "imp_best96"]:
                diffs = [r[k] - rows[(r["state"], "S3_placement_perp")][k] for r in rs
                         if k in r and (r["state"], "S3_placement_perp") in rows]
                if diffs and sname != "S3_placement_perp":
                    a[f"{k}_minus_S3_mean"] = float(np.mean(diffs))
                    a[f"{k}_minus_S3_sem"] = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else None
                    a[f"{k}_wins_vs_S3"] = f"{int(np.sum(np.array(diffs) > 0))}/{len(diffs)}"
            agg[f"{typ}/{sname}"] = a
    out = {"goals": GOALS, "value_fn": "lyapunov (improvement = v0 - v1, >0 better)",
           "resamples": R, "aggregate": agg,
           "per_state": [rows[k] for k in sorted(rows)]}
    p = EXP / "results/sampler_study.json"
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
    for k, a in agg.items():
        print(f"{k:32s} n={a['n_states']:2d} null={a.get('null_frac', float('nan')):.2f} ge3={a.get('ge3_frac', float('nan')):.2f} "
              f"moved={a.get('cubes_moved_mean', float('nan')):.2f} best96={a.get('imp_best96', float('nan')):.4f} "
              f"b16={a['best_of_16']:.4f} b8={a['best_of_8']:.4f} mean={a.get('imp_mean', float('nan')):.4f} "
              f"L1={a.get('outcome_L1_pairwise_mean', float('nan')):.3f} tsim={a.get('t_sim_s', float('nan')):.0f}")


if __name__ == "__main__":
    main()
