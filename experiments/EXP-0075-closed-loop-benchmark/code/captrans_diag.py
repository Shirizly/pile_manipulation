"""EXP-0075 captrans diagnostic (2026-10-11, results/captrans/diag.json): can the crowding-transport term steer the choice at all? On visited states of the
ct_w10 / ct_w15 captrans_captrans cells (pushes 3, 6, 9), score the capacity-transport + fetch proposals with the model under the current objective and under
captrans; report the spread of the crowd term across candidates vs the spread of the base cost, the Spearman correlation of the two objectives' costs, how often
the argmin changes, and the excess / off-slot parts. usage: PYTHONPATH=. python -u captrans_diag.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
import wide_planner as wp
from closed_loop import MODELS
from batched_closed_loop import goal_mask, goal_dist
from captrans import CrowdTransport, captrans_candidates, capacity_slots
from model.retrieval.frame import yaw_from_quat

R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/captrans"; model = MODELS["ens2bal"](); out = {}
for W in (10, 15):
    p = R / f"w{W}_H1.json"
    if not p.exists():
        continue
    eps = [e for e in json.loads(p.read_text())["episodes"] if e["cell"] == f"ct_w{W}_H1_captrans_captrans" and e.get("complete")]
    rows = []
    for e in eps[::2]:
        g = e["goal"]; mask = torch.from_numpy(goal_mask(g)); dist = goal_dist(g); sl = capacity_slots(g)
        for k in (3, 6, 9):
            S = torch.tensor(e["states"][k]).float(); Q = torch.tensor(e["quats"][k]).float(); P = torch.cat([S, Q], 1)
            xy = S[:, :2].to(wp.DEV); cand = torch.cat([captrans_candidates(xy, sl)[0], wp.fetch_candidates(xy, mask)]); cand = cand[torch.randperm(len(cand), device="cpu")[:512].to(wp.DEV)]
            leg = wp.make_legal_scorer(S[:, :2], yaw_from_quat(Q)); cand, _, ok = leg(cand); cand = cand[ok.to(cand.device)]
            if len(cand) < 10:
                continue
            crowd = CrowdTransport(S[:, :2].numpy(), sl, mask.numpy(), None)
            ob = wp.SeqObjective(model, P, dist, mask); cb = ob.cost(cand[:, None]).cpu().numpy()
            oc = wp.SeqObjective(model, P, dist, mask, crowd=crowd); cc = oc.cost(cand[:, None]).cpu().numpy()
            preds = model.predict(cand[:, None, :2], cand[:, None, 2:])[-1]; ex, off = crowd.parts(preds); ex, off = ex.cpu().numpy(), off.cpu().numpy(); ct = ex + off
            ct0 = float(sum(x for x in crowd.parts(model.occ0)))
            rows.append(dict(goal=g, start=e["start"], push=k, n=int(len(cand)), std_base=float(cb.std()), std_crowd=float(ct.std()), range_crowd=float(ct.max() - ct.min()), spearman=float(spearmanr(cb, cc)[0]),
                             argmin_same=bool(cb.argmin() == cc.argmin()), crowd_at_base_argmin=float(ct[cb.argmin()] - ct0), crowd_at_crowd_argmin=float(ct[cc.argmin()] - ct0), base_gap=float(cb[cc.argmin()] - cb.min()),
                             excess_max=float(ex.max()), off_mean=float(off.mean()), top8_overlap=float(len(set(np.argsort(cb)[:8]) & set(np.argsort(cc)[:8])) / 8)))
    A = {k: float(np.mean([r[k] for r in rows])) for k in ("std_base", "std_crowd", "range_crowd", "spearman", "argmin_same", "crowd_at_base_argmin", "crowd_at_crowd_argmin", "base_gap", "excess_max", "off_mean", "top8_overlap")}
    out[f"w{W}"] = dict(n_states=len(rows), mean=A, rows=rows); print(f"w{W}: {len(rows)} states", {k: round(v, 4) for k, v in A.items()}, flush=True)
(R / "diag.json").write_text(json.dumps(out, indent=1))
