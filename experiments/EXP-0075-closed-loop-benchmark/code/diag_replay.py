"""EXP-0075 diagnosis D2 + D3 (no simulator): replay the recorded closed-loop episodes.
D2  at every decision step k of a recorded episode, take the TRUE state k and the pushes that were actually executed from k on; ask the ensemble model to predict the state after j = 1..4 of them
    (own prediction fed back) and compare the predicted change of the planner's objective value with the true one (true state = soft truth occ_for_scoring). value = lyapunov - 1.0 * in-goal fraction.
    Reported per source cell and j: mean (pred - true) [negative = optimistic], RMSE, Spearman over decisions, and the same restricted to decisions in the first 5 steps.
D3  first-push sacrifice: at step 1 every cell starts from the same 32 states; compare the TRUE value change of the first executed push across cells (paired), and the model's prediction of it.
usage: PYTHONPATH=. python -u diag_replay.py cells...   -> results/diag_replay.json / .md"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "experiments/EXP-0075-closed-loop-benchmark/code"); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import wide_planner as wp
from analyse import value_curve
from batched_closed_loop import goal_mask, goal_dist
from simple_mpc.learned_mpc import lyap
from scipy.stats import spearmanr
R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"; RES = "experiments/EXP-0075-closed-loop-benchmark/results/"
model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
cells_want = sys.argv[1:] or ["cem1_Lfull", "cem2_Lfull", "cem4_Lfull", "cem1_L20", "cem2_L20", "cem4_L20"]
eps = []
for f in ("main_L20.json", "main_Lfull.json"):
    eps += [e for e in json.load(open(RES + f))["episodes"] if e.get("complete") and e["cell"] in cells_want]
MASK = {}; DIST = {}


def val_hard(occ, goal):   # planner's own value of a (hard/pasted) 64x64 occupancy
    f = occ.reshape(occ.shape[0], -1); return (lyap(occ, DIST[goal]) - (f * MASK[goal].reshape(1, -1)).sum(1) / f.sum(1)).cpu().numpy()


out = {}; rows = {}
for e in eps:
    g = e["goal"]
    if g not in DIST:
        DIST[g] = goal_dist(g).to(wp.DEV); MASK[g] = torch.from_numpy(goal_mask(g)).float().to(wp.DEV)
    true_v = value_curve(e["states"], g)                              # (11,) true value after every push, soft truth
    S = [torch.cat([torch.tensor(e["states"][t]), torch.tensor(e["quats"][t])], 1).float() for t in range(len(e["states"]))]
    A = torch.tensor(e["actions"]).float()                           # executed (legalised) pushes
    r = rows.setdefault(e["cell"], {j: [] for j in (1, 2, 3, 4)})
    for k in range(len(A)):
        J = min(4, len(A) - k)
        model.begin(S[k]); acts = A[k:k + J][None].to(wp.DEV); preds = model.predict(acts[..., :2], acts[..., 2:])
        v0 = val_hard(model.occ0, g)[0]
        for j in range(1, J + 1):
            r[j].append((k, float(val_hard(preds[j - 1], g)[0] - v0), float(true_v[k + j] - true_v[k])))
for c, r in rows.items():
    out[c] = {}
    for j, v in r.items():
        a = np.array(v); p, t = a[:, 1], a[:, 2]; early = a[:, 0] < 5
        out[c][j] = dict(n=len(a), bias=float((p - t).mean()), rmse=float(np.sqrt(((p - t) ** 2).mean())), spearman=float(spearmanr(p, t)[0]), true_mean=float(t.mean()), pred_mean=float(p.mean()),
                         bias_early=float((p[early] - t[early]).mean()), spearman_early=float(spearmanr(p[early], t[early])[0]))
# D3: first push, paired over (goal, start) at step 1
first = {}
for e in eps:
    tv = value_curve(e["states"], e["goal"]); first.setdefault(e["cell"], {})[(e["goal"], e["start"])] = float(tv[1] - tv[0])
d3 = {c: float(np.mean(list(v.values()))) for c, v in first.items()}
json.dump(dict(d2=out, d3_true_first_push_dvalue=d3), open(RES + "diag_replay.json", "w"), indent=1)
md = ["# D2: model prediction of the executed pushes from the TRUE state, j pushes ahead (value = lyapunov - in-goal fraction; negative bias = model over-promises)\n", "| source cell | j | n | pred change | true change | bias (pred-true) | RMSE | Spearman | bias (steps 0-4) | Spearman (steps 0-4) |", "|---|---|---|---|---|---|---|---|---|---|"]
for c in sorted(out):
    for j in (1, 2, 3, 4):
        o = out[c][j]; md.append(f"| {c} | {j} | {o['n']} | {o['pred_mean']:+.3f} | {o['true_mean']:+.3f} | {o['bias']:+.3f} | {o['rmse']:.3f} | {o['spearman']:+.2f} | {o['bias_early']:+.3f} | {o['spearman_early']:+.2f} |")
md += ["\n# D3: TRUE value change of the first executed push (same 32 start states in every cell; lower = more immediate progress)\n", "| cell | mean true dvalue of push 1 |", "|---|---|"] + [f"| {c} | {v:+.4f} |" for c, v in sorted(d3.items())]
open(RES + "diag_replay.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
