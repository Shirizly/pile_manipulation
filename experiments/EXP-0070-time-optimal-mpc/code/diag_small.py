"""EXP-0070 RUN-0008: WHY does minimising J = t(a) + etc(V_s + dV_model) - etc(V_s) pick short do-little pushes?
Offline (model only, no Genesis), on recorded visited states.

Part 1 -- per state (pilot_v0 quadrant states of the lyap / time / time_lin cells, pilot_v0b lyap states of all
7 goals; k in {1, 2, 4}; plate pose = end of the previous executed push):
  * a bank of 8192 pushes (4096 uniform in the tray, 4096 "pile-aimed": start 5-20 mm behind a random cube,
    random heading), projected to 20-70 mm exactly like plan();
  * CEM on J (0.5 s, pop 1024, elite 0.25 -- the v0 `time` cell) -> a_J; CEM on model dV (the lyap planner)
    -> a_V; both scored under BOTH objectives;
  * decomposition J = t + d_etc for a_J, a_V, the bank's J-argmin and dV-argmin;
  * Bellman residual of the lyap planner's push under the map: r = t_plan + t(a_V) + d_etc(a_V) (0 if etc
    were consistent with what the planner's own model says one push achieves);
  * break-even etc scale: the factor lambda by which etc would have to be multiplied for a_V to beat a_J
    (J_lambda = t + lambda * d_etc).
Part 2 -- calibration: for every executed push of pilot_v0 / v0b / v1 / b01, the model's dV (as TimeObjective
reads it) vs the TRUE dV recorded at run time, binned by push length and by the cell that chose it.
-> results/diag_small.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_dist
from simple_mpc.adapters import make_occ_adapter
import simple_mpc.learned_mpc as LM
from time_mpc import TimeObjective, load_etc, etc_torch
from action_time import action_time, HOME

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
T_PLAN = 0.5
dev = "cuda"


def bank(parts, gen, n=4096):
    lo, hi = -0.064 + LM.XY_MARGIN, 0.064 - LM.XY_MARGIN
    th = torch.rand(2 * n, generator=gen) * 2 * math.pi
    L = 0.02 + 0.05 * torch.rand(2 * n, generator=gen)
    d = torch.stack([torch.cos(th), torch.sin(th)], 1)
    s_uni = lo + (hi - lo) * torch.rand(n, 2, generator=gen)
    xy = parts[:, :2]
    c = xy[torch.randint(len(xy), (n,), generator=gen)]
    back = 0.005 + 0.015 * torch.rand(n, 1, generator=gen)
    lat = (torch.rand(n, 1, generator=gen) - 0.5) * 0.02
    perp = torch.stack([-d[n:, 1], d[n:, 0]], 1)
    s_pile = c - back * d[n:] + lat * perp
    s = torch.cat([s_uni, s_pile]).clamp(lo, hi)
    A = torch.cat([s, s + L[:, None] * d], 1)
    return LM.project_push(A)[0]


def states():
    out = []
    for tag, cells, goals in (("pilot_v0", ("lyap", "time", "time_lin"), None), ("pilot_v0b", ("lyap",), None)):
        d = json.loads((HERE / f"results/{tag}.json").read_text())
        for e in d["episodes"]:
            if not e.get("complete") or e["cell"] not in cells:
                continue
            for k in (1, 2, 4):
                out.append(dict(src=tag, cell=e["cell"], goal=e["goal"], start=e["start"], k=k,
                                parts=torch.cat([torch.tensor(e["states"][k]), torch.tensor(e["quats"][k])], -1).float(),
                                p_prev=e["actions"][k - 1][2:4], executed=e["actions"][k], true_dv_exec=e["true_dv"][k]))
    return out


def evaluate(obj, A):
    with torch.no_grad():
        obj.raw = True; dv = obj(A.to(dev).float()); obj.raw = False
        J = obj(A.to(dev).float())
    t = action_time(A.to(dev).float(), obj.p_prev, False)
    return dv.cpu().numpy(), J.cpu().numpy(), t.cpu().numpy()


def part1(ad, etc):
    rows = []
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    S = states()
    t0 = time.time()
    for i, s in enumerate(S):
        g = s["goal"]; dist = goal_dist(g)
        ctx = dict(mode="time", etc=etc[g], p_prev=s["p_prev"], first=False, t_plan=T_PLAN)
        obj = TimeObjective(ad, s["parts"], dist, ctx=ctx)
        B = bank(s["parts"], gen)
        dvB, JB, tB = evaluate(obj, B)
        aJ = LM.plan(obj, B[:1024], "cem", T_PLAN, cem_pop=1024, cem_elite_frac=0.25)["action"]
        obj.raw = True
        aV = LM.plan(obj, B[:1024], "cem", T_PLAN, cem_pop=1024, cem_elite_frac=0.25)["action"]
        obj.raw = False
        ex = torch.tensor(s["executed"])[None]
        picks = torch.stack([aJ, aV, B[int(JB.argmin())], B[int(dvB.argmin())], ex[0]])
        dvP, JP, tP = evaluate(obj, picks)
        L = (picks[:, 2:] - picks[:, :2]).norm(dim=1).numpy()
        tr = (picks[:, :2] - torch.tensor(s["p_prev"])).norm(dim=1).numpy()
        names = ["cemJ", "cemV", "bankJ", "bankV", "executed"]
        rec = dict(src=s["src"], cell=s["cell"], goal=g, start=s["start"], k=s["k"], Vs=float(obj.Vs),
                   etc_slope=float(torch.autograd.functional.jacobian(lambda v: etc_torch(etc[g], v), obj.Vs[None].float())[0, 0]),
                   **{f"{n}_{q}": float(v[j]) for j, n in enumerate(names)
                      for q, v in (("dv", dvP), ("J", JP), ("t", tP), ("L", L), ("transit", tr))},
                   true_dv_exec=s["true_dv_exec"],
                   bank_frac_J_neg=float((JB < 0).mean()), bank_frac_dv_neg=float((dvB < -1e-3).mean()),
                   bank_min_J=float(JB.min()))
        dE = lambda n: rec[f"{n}_J"] - rec[f"{n}_t"]                     # d_etc part (s, negative = progress)
        rec["bellman_r_cemV"] = T_PLAN + rec["cemV_J"]
        # lambda at which cemV ties cemJ: t_V + l dE_V = t_J + l dE_J
        den = dE("cemV") - dE("cemJ")
        rec["lambda_break_even"] = float((rec["cemJ_t"] - rec["cemV_t"]) / den) if den < -1e-9 else None
        rows.append(rec)
        if i % 25 == 0:
            print(f"{i}/{len(S)} {time.time() - t0:.0f}s", flush=True)
    return rows


def part2(ad):
    out = []
    for tag in ("pilot_v0", "pilot_v0b", "pilot_v1", "pilot_b01"):
        d = json.loads((HERE / f"results/{tag}.json").read_text())
        for e in d["episodes"]:
            if not e.get("complete"):
                continue
            dist = goal_dist(e["goal"])
            for k in range(len(e["actions"])):
                parts = torch.tensor(e["states"][k]).float()
                obj = LM.ModelObjective(ad, parts, dist)
                with torch.no_grad():
                    pdv = float(obj(torch.tensor([e["actions"][k]]).float().to(dev))[0])
                a = e["actions"][k]
                out.append(dict(src=tag, cell=e["cell"], goal=e["goal"], k=k, L=math.hypot(a[2] - a[0], a[3] - a[1]),
                                pred=pdv, true=e["true_dv"][k], V=e["values"][k]))
    return out


def summarise(rows, cal):
    S = {}
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if isinstance(rows[0][k], (int, float)) and rows[0][k] is not None}
    def grp(mask):
        m = {}
        for n in ("cemJ", "cemV", "bankJ", "bankV", "executed"):
            m[n] = {q: float(np.nanmean(R[f"{n}_{q}"][mask])) for q in ("dv", "J", "t", "L", "transit")}
        m["frac_cemV_better_on_J"] = float(np.mean(R["cemV_J"][mask] < R["cemJ_J"][mask]))
        m["frac_bankJ_better_than_cemJ"] = float(np.mean(R["bankJ_J"][mask] < R["cemJ_J"][mask] - 1e-6))
        m["frac_minJ_positive_cemJ"] = float(np.mean(R["cemJ_J"][mask] > 0))
        m["bellman_r_cemV_mean"] = float(np.mean(R["bellman_r_cemV"][mask]))
        lam = np.array([r["lambda_break_even"] if r["lambda_break_even"] is not None else np.nan for r in rows])[mask]
        m["lambda_break_even_median"] = float(np.nanmedian(lam)) if np.any(~np.isnan(lam)) else None
        m["n"] = int(mask.sum())
        return m
    fam = np.array(["quadrant" if r["goal"].startswith("quad") else ("two_squares" if r["goal"] == "two_squares" else "letter") for r in rows])
    S["ALL"] = grp(np.ones(len(rows), bool))
    for f in ("quadrant", "two_squares", "letter"):
        S[f] = grp(fam == f)
    C = {}
    pr, tr, L = (np.array([c[k] for c in cal]) for k in ("pred", "true", "L"))
    for lo, hi in ((0.0, 0.03), (0.03, 0.045), (0.045, 0.06), (0.06, 0.08)):
        m = (L >= lo) & (L < hi)
        if m.sum() > 5:
            C[f"L{lo*1000:.0f}-{hi*1000:.0f}mm"] = dict(n=int(m.sum()), pred=float(pr[m].mean()), true=float(tr[m].mean()),
                                                     bias=float((pr[m] - tr[m]).mean()), corr=float(np.corrcoef(pr[m], tr[m])[0, 1]),
                                                     ratio_pred_over_true=float(pr[m].sum() / tr[m].sum()))
    for cell in sorted({c["cell"] for c in cal}):
        m = np.array([c["cell"] == cell for c in cal])
        C[f"cell_{cell}"] = dict(n=int(m.sum()), pred=float(pr[m].mean()), true=float(tr[m].mean()), bias=float((pr[m] - tr[m]).mean()),
                                 mean_L=float(L[m].mean()))
    return S, C


def main():
    ad = make_occ_adapter("nfd_3ch_randlen", dev, "corner")
    etc = load_etc(HERE / "results/etc_map_v0.json")
    LM.plan(LM.ModelObjective(ad, torch.zeros(20, 3), goal_dist("quadrant_0")), torch.zeros(8, 4) + 0.01, "cem", 0.1)
    rows = part1(ad, etc)
    p = HERE / "results/diag_small_rows.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(rows)); os.replace(tmp, p)
    cal = part2(ad)
    S, C = summarise(rows, cal)
    out = HERE / "results/diag_small.json"; tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(dict(by_family=S, calibration=C), indent=1)); os.replace(tmp, out)
    print(json.dumps(dict(by_family=S, calibration=C), indent=1))


if __name__ == "__main__":
    main()
