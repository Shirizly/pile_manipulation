"""block C + D: H=4 plans, time-calibrated budgets 1/3/10 s, models x planners x 10 tasks; one JSON per plan (atomic).  usage: bs_plan_h4.py shard nshards"""
import sys, json, os
from bs_lib import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
jobs = []
for B in (1, 3, 10):
    for key in MODELS:
        for planner in ("cem", "cem_gd") + (("rand_gd",) if key in ("ens128", "vanilla64") else ()):
            for ti, (g, s) in enumerate(TASKS):
                jobs.append((key, planner, B, ti, g, s))
for ji, (key, planner, B, ti, g, s) in enumerate(jobs):
    f = BS / "plans" / f"{key}__{planner}__B{B}__{g}_{s}.json"
    if ji % nsh != shard or f.exists(): continue
    r = plan(key, START(s), g, 4, B, planner, 1000 + ti); r.update(goal=g, start=s); json.dump(r, open(str(f) + ".tmp", "w")); os.replace(str(f) + ".tmp", f)
    print(ji, len(jobs), key, planner, B, g, s, round(r["cost"], 3), f"{r['plan_time_s']:.1f}s (budget {B})", flush=True)
