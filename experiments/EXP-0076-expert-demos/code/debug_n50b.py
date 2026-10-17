import sys, time, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim, legalize
from transforms.functional import action_to_pose
sc = [s for s in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"] if s["n"] == 50]
pick = [s for s in sc if s["id"].endswith(("-000", "-001"))]   # 2 per stratum = 18
E = len(pick); sim = Sim(50, E)
st = torch.stack([torch.tensor(s["state0"], device="cpu").float() for s in pick]); bank = sim.bank(st, 256)
for rep in range(2):
    acts = []
    for i in range(E):
        a, _, ok = legalize(bank[i].cuda(), st[i]); acts.append(a[ok][rep].cpu())
    # one env at a time (E envs all set, but only measure per-env cost by running the batch with every env doing env i's push)
    sim.set(st); gs = sim.gs; A = torch.stack(acts)
    sx, sy, ex, ey, ang = action_to_pose(A.to(gs.device)); z = torch.full_like(sx, float(sim.sim._operation_height))
    t0 = time.time(); sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); t1 = time.time(); sim.sim.update_material_state(); t2 = time.time()
    print(f"batch E{E} rep{rep}: execute {t1 - t0:.1f}s settle {t2 - t1:.1f}s; push lengths mm {[round(float((a[2:] - a[:2]).norm()) * 1000) for a in acts]}", flush=True)
for i in [0, 6, 12]:   # per-init single-env cost: put that env's state + push everywhere
    for j in range(3):
        k = i + j * 2
    pass
for i, s in enumerate(pick):
    if i % 2: continue
    sim.set(st[i:i + 1].expand(E, -1, -1)); A = acts[i][None].expand(E, -1)
    sx, sy, ex, ey, ang = action_to_pose(A.to(gs.device)); z = torch.full_like(sx, float(sim.sim._operation_height))
    t0 = time.time(); sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); t1 = time.time(); sim.sim.update_material_state(); t2 = time.time()
    print(f"{s['id']}: execute {t1 - t0:.1f}s settle {t2 - t1:.1f}s len {float((acts[i][2:] - acts[i][:2]).norm()) * 1000:.0f}mm", flush=True)
