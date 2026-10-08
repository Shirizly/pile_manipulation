"""block A: greedy H=1 open loop on the model's predicted states (ens128), pools 1,280 and 10,000 (+ GD 150 steps x 24 on the last push), 10 tasks; plans saved like block C."""
import sys, json, os
from bs_lib import *
MK = os.environ.get('BS_GREEDY_MODEL', 'ens128')
for pool in (1280, 10000):
    for ti, (g, s) in enumerate(TASKS[:10 if (pool == 1280 or MK == 'lf') else 3]):
        f = BS / "plans" / f"{MK}__greedy_pool{pool}__Bna__{g}_{s}.json"
        if f.exists(): continue
        r = greedy_open_loop(MK, START(s), g, pool, seed=2000 + ti); r.update(goal=g, start=s, B=None); json.dump(r, open(str(f) + ".tmp", "w")); os.replace(str(f) + ".tmp", f)
        print(pool, g, s, round(r["cost"], 3), f"{r['plan_time_s']:.0f}s", flush=True)
