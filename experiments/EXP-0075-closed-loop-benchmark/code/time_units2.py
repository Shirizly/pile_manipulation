"""EXP-0075: GD step time vs batch size, proposal time, CEM bookkeeping overhead -> results/timing_units2.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
import wide_planner as wp
from intensive import propose_from_occ, legal_first, rand_later, proj, R74
from time_units import M, sync
OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); S0 = rows.states[(rows.slate_idx == 40).nonzero()[0, 0]].float(); goal = "letter_T_w20"; res = {}
for name in ("ens128 (zoom+vanilla)", "zoom128", "vanilla128", "zoom64 (f8)", "vanilla64 (f4)"):
    model = wp.WideEns(M[name](), balance=True); obj = wp.SeqObjective(model, S0, goal_dist(goal), torch.from_numpy(goal_mask(goal)), 1.0, chunk=256); r = {}
    sync(); t0 = time.time(); first = legal_first(propose_from_occ(model.occ0[0], 60000), S0); sync(); r["propose_60000_s"] = time.time() - t0; r["n_first"] = len(first)
    t0 = time.time(); model.begin(S0); sync(); r["begin_state_s"] = time.time() - t0
    for H in (1, 4):
        for n in (4, 8, 24, 64):
            seqs = torch.cat([first[torch.randint(0, len(first), (n,), device=wp.DEV)][:, None], rand_later(first, n, H)], 1) if H > 1 else first[torch.randint(0, len(first), (n,), device=wp.DEV)][:, None]
            x = seqs.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
            for it in range(6):
                if it == 2: sync(); t0 = time.time()
                opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); pr = model._predict(xp[..., :2], xp[..., 2:], grad=True)[-1]; obj._value(pr).sum().backward(); opt.step()
            sync(); r[f"gd_step_H{H}_n{n}_ms"] = (time.time() - t0) / 4 * 1000
    res[name] = r; print(name, {k: round(v, 3) for k, v in r.items()}, flush=True); del model, obj; torch.cuda.empty_cache()
json.dump(res, open(OUT / "timing_units2.json", "w"), indent=1)
