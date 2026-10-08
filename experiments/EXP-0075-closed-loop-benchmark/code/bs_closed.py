"""blocks B + E: closed-loop H=1 episodes (4 pushes, re-planned from the TRUE simulator state before every push), 10 tasks per config, saving every transition (states before/after, planned and legalised actions, values).
 eq   : per push budget B/4 (B = 1,3,10 s), planner cem_gd, models = all 5   (equal planning time per 4 pushes as one H=4 call)
 ceil : ens128, pool 1,280 / 10,000 + GD 150 x 24 per push (H=1 ceiling)
usage: bs_closed.py eq|ceil [shard nshards]"""
import sys
from bs_lib import *
mode = sys.argv[1]; shard, nsh = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
cfgs = [("ens128", f"pool{p}") for p in (1280, 10000)] if mode == "ceil" else [(k, B) for B in (1, 3, 10) for k in MODELS]
groups = [cfgs[i:i + 3] for i in range(0, len(cfgs), 3)] if mode == "eq" else [cfgs]
sim = Sim(32)
for gi, grp in enumerate(groups):
    tag = f"closed_{mode}_{gi}"
    if gi % nsh != shard or (BS / f"replay_{tag}.json").exists(): continue
    eps = [(c, ti, g, s) for c in grp for ti, (g, s) in enumerate(TASKS[:5] if c[1] == 'pool10000' else TASKS)]; n = len(eps); t0 = time.time()
    p = sim.reset([START(s) for _, _, _, s in eps]); goals = [g for _, _, g, _ in eps] + [eps[0][2]] * (32 - n); vals = [true_value(p, goals)]; parts = [p.numpy()]; planned, legal, shifts, preds, ptimes = [], [], [], [], []
    for j in range(4):
        acts = torch.zeros(32, 4); pp, tt = [], []
        for k, ((key, cf), ti, g, s) in enumerate(eps):
            t1 = time.time()
            if mode == "ceil": r = greedy_open_loop(key, p[k], g, int(cf[4:]), seed=3000 + 10 * ti + j, H=1)
            else: r = plan(key, p[k], g, 1, cf / 4, "cem_gd", 3000 + 10 * ti + j)
            acts[k] = torch.tensor(r["seq"][0]); pp.append(r["pred_per_push"][0]); tt.append(time.time() - t1)
        acts[n:] = acts[0]; a, sh, ok, p = sim.push(acts); planned.append(acts.cpu().numpy()); legal.append(a.cpu().numpy()); shifts.append(sh); preds.append(pp); ptimes.append(tt); vals.append(true_value(p, goals)); parts.append(p.numpy())
        print(tag, "push", j + 1, f"{time.time() - t0:.0f}s", flush=True)
    vals = np.stack(vals, 1); out = []
    for k, ((key, cf), ti, g, s) in enumerate(eps):
        out.append(dict(goal=g, start=s, model=key, planner="h1_closed_" + mode, B=cf if mode == "eq" else None, pool=cf if mode == "ceil" else None, true_terminal=float(vals[k, -1] - vals[k, 0]), true_per_push=(vals[k, 1:] - vals[k, 0]).tolist(),
                        pred_step_values=[preds[j][k] for j in range(4)], plan_time_s=float(sum(ptimes[j][k] for j in range(4))), shifted_pushes=int(sum(sh[k] > 0 for sh in shifts))))
    np.savez_compressed(BS / f"states_{tag}.npz", parts=np.stack(parts, 1)[:n], planned=np.stack(planned, 1)[:n], legalised=np.stack(legal, 1)[:n], shifts=np.stack(shifts, 1)[:n], vals=vals[:n])
    json.dump(out, open(BS / f"replay_{tag}.json.tmp", "w")); os.replace(BS / f"replay_{tag}.json.tmp", BS / f"replay_{tag}.json")
sim.close()
