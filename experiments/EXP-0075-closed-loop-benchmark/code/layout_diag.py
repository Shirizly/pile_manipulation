"""EXP-0075 LAYOUT-term diagnostic (2026-10-11, results/layout/diag.json; adapted from captrans_diag.py): on visited states of the ct_w10 / ct_w15 cur_fetch H=1 cells (pushes 3/6/9) score OT + fetch proposals under the current objective and
current + lam x LayoutTransport (lam 1/2/4); report term magnitude vs base-cost spread, Spearman, argmin-same, top-8 overlap. Cost pilot on ranking only (no closed-loop outcome).
(original docstring:) EXP-0075 captrans diagnostic (2026-10-11, results/captrans/diag.json): can the crowding-transport term steer the choice at all? On visited states of the
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
from layout_term import LayoutTransport
from simple_mpc.value_functions import uniform_layout
from model.retrieval.frame import yaw_from_quat

R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/captrans"; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/layout"; OUT.mkdir(exist_ok=True); model = MODELS["ens2bal"](); out = {}; LAMS = (1.0, 2.0, 4.0)
for W in (10, 15):
    p = R / f"w{W}_H1.json"
    if not p.exists():
        continue
    eps = [e for e in json.loads(p.read_text())["episodes"] if e["cell"] == f"ct_w{W}_H1_cur_fetch" and e.get("complete")]
    rows = []
    for e in eps[::2]:
        g = e["goal"]; mask = torch.from_numpy(goal_mask(g)); dist = goal_dist(g); sl = uniform_layout(mask.numpy(), 20)
        for k in (3, 6, 9):
            S = torch.tensor(e["states"][k]).float(); Q = torch.tensor(e["quats"][k]).float(); P = torch.cat([S, Q], 1)
            xy = S[:, :2].to(wp.DEV); cand = torch.cat([wp.ot_candidates(xy, mask, torch.from_numpy(sl).float())[0], wp.fetch_candidates(xy, mask)]); cand = cand[torch.randperm(len(cand), device="cpu")[:512].to(wp.DEV)]
            leg = wp.make_legal_scorer(S[:, :2], yaw_from_quat(Q)); cand, _, ok = leg(cand); cand = cand[ok.to(cand.device)]
            if len(cand) < 10:
                continue
            ob = wp.SeqObjective(model, P, dist, mask); cb = ob.cost(cand[:, None]).cpu().numpy()
            preds = model.predict(cand[:, None, :2], cand[:, None, 2:])[-1]
            for lam in LAMS:
                lt = LayoutTransport(sl, mask.numpy(), lam=lam); ct = lt(preds).detach().cpu().numpy(); ct0 = float(lt(model.occ0)[0]); cc = cb + ct
                rows.append(dict(lam=lam, goal=g, start=e["start"], push=k, n=int(len(cand)), std_base=float(cb.std()), std_crowd=float(ct.std()), range_crowd=float(ct.max() - ct.min()), term_now=ct0, base_mean_abs=float(np.abs(cb).mean()),
                                 spearman=float(spearmanr(cb, cc)[0]), argmin_same=bool(cb.argmin() == cc.argmin()), base_gap=float(cb[cc.argmin()] - cb.min()), top8_overlap=float(len(set(np.argsort(cb)[:8]) & set(np.argsort(cc)[:8])) / 8)))
    for lam in LAMS:
        rl = [r for r in rows if r["lam"] == lam]; A = {k: float(np.mean([r[k] for r in rl])) for k in ("std_base", "std_crowd", "range_crowd", "term_now", "spearman", "argmin_same", "base_gap", "top8_overlap")}
        out[f"w{W}_lam{lam}"] = dict(n_states=len(rl), mean=A, rows=rl); print(f"w{W} lam {lam}: {len(rl)} states", {k: round(v, 4) for k, v in A.items()}, flush=True)
(OUT / "diag.json").write_text(json.dumps(out, indent=1))
