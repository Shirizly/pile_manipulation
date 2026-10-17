"""EXP-0070 RUN-0007: can an existing learned model rank CHAINED pushes (rotate in place about the blade
centre, then sweep from the current plate pose; the plate never lifts) as well as ordinary free pushes?
Uses the Genesis outcomes saved by chain_probe.py --mode chain (results/chain_probe_states/*.pt).

Per saved state and candidate set (32 free / 32 chained), with the broad nfd_3ch_randlen:
  1-step : predict from the TRUE plate-down state (occ_from_particles(base)).
  2-step : predict from the model's own prediction of that state (rolled from s_{k-1} through a_{k-1}),
           i.e. what a planner composing a two-segment sweep without an intermediate observation sees.
Metrics (higher better unless noted): Spearman(pred dV, true dV); capture = true dV of the model's pick /
best true dV in the set (a 32-candidate slateN analogue; 1 = picked the best); |pred - true| dV error
(lower better). The model's dV is measured against its own input representation (as ModelObjective does);
true dV uses soft scoring (occ_for_scoring). -> results/chain_model_eval.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_dist
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, occ_for_scoring
from simple_mpc.learned_mpc import lyap

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "nfd_3ch_randlen"


def main():
    dev = "cuda"
    ad = make_occ_adapter(MODEL, dev, "corner")
    rows = []
    for f in sorted((HERE / "results/chain_probe_states").glob("*.pt")):
        d = torch.load(f, weights_only=False); dist = goal_dist(d["goal"]).to(dev)
        base_true = occ_from_particles(d["base"][None, :, :3].float(), dev)
        prev = occ_from_particles(d["prev"][None, :, :3].float(), dev)
        with torch.no_grad():
            base_pred = ad.predict_step(prev, d["a_prev"][None].float().to(dev))
        V0_true = float(lyap(occ_for_scoring(d["base"][None, :, :3].float().to(dev)), dist)[0])
        rec = dict(file=f.name, goal=d["goal"])
        for kind in ("free", "chain"):
            A = d[kind].float().to(dev)
            tdv = (lyap(occ_for_scoring(d[f"after_{kind}"][:, :, :3].float().to(dev)), dist) - V0_true).cpu().numpy()
            for src, occ0 in (("1step", base_true), ("2step", base_pred)):
                with torch.no_grad():
                    p1 = ad.predict_step(occ0.expand(len(A), -1, -1).contiguous(), A)
                pdv = (lyap(p1, dist) - lyap(occ0, dist)[0]).cpu().numpy()
                pick = int(np.argmin(pdv)); best = float(tdv.min())
                rec[f"{kind}|{src}"] = dict(spearman=float(spearmanr(pdv, tdv).correlation),
                                            capture=float(tdv[pick] / best) if best < 0 else float("nan"),
                                            mae=float(np.abs(pdv - tdv).mean()), best_true=best)
        rows.append(rec)
    out = dict(model=MODEL, n_states=len(rows), per_state=rows, summary={})
    for key in ("free|1step", "free|2step", "chain|1step", "chain|2step"):
        for m in ("spearman", "capture", "mae"):
            v = np.array([r[key][m] for r in rows], float)
            out["summary"][f"{key}|{m}"] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), n=int(np.sum(~np.isnan(v))))
    for m in ("spearman", "capture", "mae"):   # paired chain - free (1-step)
        dlt = np.array([r["chain|1step"][m] - r["free|1step"][m] for r in rows], float)
        out["summary"][f"chain_minus_free_1step|{m}"] = dict(mean=float(np.nanmean(dlt)), sd=float(np.nanstd(dlt)))
    p = HERE / f"results/chain_model_eval_{MODEL}.json"; tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
    for k, v in out["summary"].items():
        print(f"{k:34s} {v['mean']:+.4f} (sd {v['sd']:.4f})")


if __name__ == "__main__":
    main()
