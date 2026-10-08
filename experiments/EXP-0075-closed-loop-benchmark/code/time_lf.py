"""EXP-0075: unit timing of the LF model (same protocol as time_units.py / time_units2.py) appended to results/timing_units.json and timing_units2.json."""
import json, time
from bs_lib import *
from time_units import sync
g, s = TASKS[0]; S0 = START(s); model = get_model("lf"); obj = make_obj(model, S0, g); first = propose_from_occ(model.occ0[0], 20000); r1, r2 = {}, {}
for H in (1, 2, 4):
    for chunk in (128, 256, 512):
        obj.chunk = chunk; N = 2048; seqs = pool_seqs(first, N, H); obj.cost(seqs[:256]); sync(); t0 = time.time(); obj.cost(seqs); sync(); r1[f"fwd_H{H}_chunk{chunk}_ms_per_seq"] = (time.time() - t0) / N * 1000
obj.chunk = 256
for H in (1, 4):
    for n in (4, 8, 24, 64):
        x = pool_seqs(first, n, H).clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
        for it in range(6):
            if it == 2: sync(); t0 = time.time()
            opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); pr = model._predict(xp[..., :2], xp[..., 2:], grad=True)[-1]; obj._value(pr).sum().backward(); opt.step()
        sync(); r2[f"gd_step_H{H}_n{n}_ms"] = (time.time() - t0) / 4 * 1000
print(r1, r2)
for f, r in (("timing_units.json", r1), ("timing_units2.json", r2)):
    d = json.load(open(RES / f)); d["LF (switched linear 64)"] = r; json.dump(d, open(RES / f, "w"), indent=1)
