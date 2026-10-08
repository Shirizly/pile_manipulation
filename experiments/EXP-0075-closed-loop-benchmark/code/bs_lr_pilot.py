"""pilot: GD learning rate for short GD phases (3 tasks, ens128 and vanilla64, budgets 1 and 3 s, cem_gd)."""
import sys, json
from bs_lib import *
out = {}
for key in ("ens128", "vanilla64"):
    for B in (1, 3):
        for lr in (2e-3, 1e-2, 3e-2):
            c = [plan(key, START(s), g, 4, B, "cem_gd", 1000 + i, lr=lr)["cost"] for i, (g, s) in enumerate(TASKS[:3])]; out[f"{key}_B{B}_lr{lr}"] = c; print(key, B, lr, np.round(c, 3).tolist(), round(float(np.mean(c)), 3), flush=True)
json.dump(out, open(RES / "bs_lr_pilot.json", "w"))
