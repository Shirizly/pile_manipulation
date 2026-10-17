"""Is a push from an identical state identical across envs of one batch? (decides whether repeated-sim robust selection is possible)"""
import sys, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from simcore import Sim, legalize
sc = [s for s in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"] if s["id"] in ("n20-scattered-disc-000", "n20-inbetween-square-001")]
E = 16; sim = Sim(20, E)
for s in sc:
    st = torch.tensor(s["state0"], device="cpu").float()
    b = sim.bank(st[None].expand(E, -1, -1).contiguous(), 64)[0]; a, _, ok = legalize(b.cuda(), st); A = a[ok][:2].cpu()
    for j in range(2):
        out = sim.step_from(st[None].expand(E, -1, -1).contiguous(), A[j][None].expand(E, -1).contiguous())
        d = (out[:, :, :2] - out[:1, :, :2]).norm(dim=-1).max(1).values * 1000
        print(s["id"], "push", j, "max cube diff vs env 0 per env (mm):", [round(float(x), 3) for x in d], flush=True)
