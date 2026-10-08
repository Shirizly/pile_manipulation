"""EXP-0075: how much CEM sampling does the gradient-refinement (GD) phase need?  Model only (zoom128+vanilla128 ensemble, mass balance), H=4, terminal objective, pile-aware proposals (as intensive.py).
Per task (goal, start): run CEM pop 10,000 x 20 iterations (210,000 evals, elite 500, as intensive.py 'cem10k'); then
  ref      benchmark CEM (pool 256, pop 256, 4 it; 1,280 evals)                       -> plan as is
  gd_ref   GD (Adam lr 2e-3, 150 steps, batch of the 24 best sequences sampled so far) started from the 24 best of the benchmark run's samples
  gd5      same, from the 24 best of the first CEM batch  (10,000 evals  ~ 5 % of 210,000)
  gd10     same, from the 24 best of the first two batches (20,000 evals ~ 10 %)
  cem      the full-budget CEM best (210,000 evals)
  gd_full  GD from the 24 best of all 210,000 samples
Saves results/gd_budget/<goal>_<start>.json (atomic per task; sequences, predicted terminal value, time).   usage: PYTHONPATH=. python -u gd_budget.py <shard> <n_shards>"""
import json, sys, time, os
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
import wide_planner as wp
from intensive import propose_from_occ, legal_first, rand_later, proj, R74, H
DEV = wp.DEV; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/gd_budget"
TASKS = [("letter_T_w20", 40), ("letter_O_w20", 41), ("letter_S_w20", 42), ("letter_X_w20", 43), ("letter_T_w20", 44), ("letter_O_w20", 45), ("letter_S_w20", 46), ("letter_X_w20", 47), ("letter_O_w20", 40), ("letter_X_w20", 41)]


def gd(obj, init, steps=150):
    x = init.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
    for _ in range(steps):
        opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); pr = obj.model._predict(xp[..., :2], xp[..., 2:], grad=True)[-1]; obj._value(pr).sum().backward(); opt.step()
    with torch.no_grad():
        xp = proj(x.detach().reshape(-1, 4)).reshape(-1, H, 4); c = obj.cost(xp); i = int(c.argmin())
    return xp[i].cpu(), float(c[i])


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2]); rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
    model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
    for ti, (goal, start) in enumerate(TASKS):
        if ti % nsh != shard or (OUT / f"{goal}_{start}.json").exists():
            continue
        torch.manual_seed(100 + ti); np.random.seed(100 + ti)
        S0 = rows.states[(rows.slate_idx == start).nonzero()[0, 0]].float(); obj = wp.SeqObjective(model, S0, goal_dist(goal), torch.from_numpy(goal_mask(goal)), 1.0, chunk=256); occ0 = model.occ0[0]
        first = legal_first(propose_from_occ(occ0, 60000), S0); res = dict(goal=goal, start=start, n_first=len(first)); t_task = time.time()
        # benchmark CEM with its samples recorded through plan_seq's top-k output
        t0 = time.time(); o = wp.plan_seq(obj, first[:4096].cpu(), "cem", H, topk=24); res["ref"] = dict(seq=o["seq"].tolist(), cost=float(o["cost"]), time_s=time.time() - t0, n_evals=o["n_evals"])
        t0 = time.time(); s, c = gd(obj, o["top_seqs"].to(DEV)[:24]); res["gd_ref"] = dict(seq=s.tolist(), cost=c, time_s=time.time() - t0, n_evals=o["n_evals"], gd_steps=150 * 24)
        print(goal, start, 'ref', round(res['ref']['cost'], 3), 'gd_ref', round(res['gd_ref']['cost'], 3), flush=True)
        # full CEM
        n_el = 500; t_cem = time.time(); pool = torch.cat([first[torch.randint(0, len(first), (10000,), device=DEV)][:, None], rand_later(first, 10000, H)], 1); cost = obj.cost(pool)
        bi = int(cost.argmin()); best, bc = pool[bi].clone(), float(cost[bi]); el = pool[cost.argsort()[:n_el]]; mean, std = el.mean(0), el.std(0).clamp_min(1e-3); S_all, C_all, curve = [pool], [cost], [bc]; t_at = {}
        for it in range(2):    # cheap variants only: first two CEM batches (full 210k-eval CEM dropped on request)
            if it > 0:
                s_ = proj((mean + std * torch.randn(10000, H, 4, device=DEV)).reshape(-1, 4)).reshape(10000, H, 4); c_ = obj.cost(s_); k = int(c_.argmin())
                if float(c_[k]) < bc: best, bc = s_[k].clone(), float(c_[k])
                el = s_[c_.argsort()[:n_el]]; mean, std = el.mean(0), el.std(0).clamp_min(1e-3); S_all.append(s_); C_all.append(c_); curve.append(bc)
            if it in (0, 1):                                  # after 1 / 2 batches: 10,000 / 20,000 evals
                name = "gd5" if it == 0 else "gd10"; SA, CA = torch.cat(S_all), torch.cat(C_all); t_cem_so_far = time.time() - t_cem
                t0 = time.time(); s, c = gd(obj, SA[CA.argsort()[:24]]); print(name, round(c, 3), flush=True); res[name] = dict(seq=s.tolist(), cost=c, time_s=time.time() - t0, cem_time_s=t_cem_so_far, n_evals=10000 * (it + 1), cem_best_cost=bc, gd_steps=150 * 24)
        res["task_time_s"] = time.time() - t_task
        json.dump(res, open(OUT / f"{goal}_{start}.tmp", "w")); os.replace(OUT / f"{goal}_{start}.tmp", OUT / f"{goal}_{start}.json")
        print(goal, start, {k: round(res[k]["cost"], 3) for k in ("ref", "gd_ref", "gd5", "gd10")}, f"{res['task_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
