import sys, time, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim, legalize
from transforms.functional import action_to_pose
ids = sys.argv[1:]
sc = {x["id"]: x for x in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"]}
pick = [sc[i] for i in ids]; E = len(pick); sim = Sim(50, E)
st = torch.stack([torch.tensor(s["state0"], device="cpu").float() for s in pick]); bank = sim.bank(st, 64)
A = []
for i in range(E):
    a, _, ok = legalize(bank[i].cuda(), st[i]); A.append(a[ok][0].cpu())
A = torch.stack(A); print("push lengths mm", [round(float((x[2:] - x[:2]).norm()) * 1000) for x in A], flush=True)
sim.set(st); gs = sim.gs; sx, sy, ex, ey, ang = action_to_pose(A.to(gs.device)); z = torch.full_like(sx, float(sim.sim._operation_height))
cnt = [0]; t0 = time.time(); last = [t0]
def on_step(*args, **kw):
    cnt[0] += 1
    if cnt[0] % 10 == 0:
        v = sim.sim._scene.rigid_solver.get_dofs_velocity() if hasattr(sim.sim._scene.rigid_solver, "get_dofs_velocity") else None
        dt = time.time() - last[0]; print(f"  step {cnt[0]} dt10 {dt:.1f}s", flush=True); last[0] = time.time()
        if dt > 40: raise RuntimeError(f"ABORT slow: {dt:.0f}s per 10 steps at step {cnt[0]}")
sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang, on_step=on_step)
t1 = time.time(); sim.sim.update_material_state(); t2 = time.time()
print(f"E{E}: execute {t1 - t0:.1f}s ({cnt[0]} steps) settle {t2 - t1:.1f}s", flush=True)
S1 = sim.state()[:E]; d = (S1[:, :, :2] - st[:, :, :2]).norm(dim=-1).max(1).values
print("max disp mm per env", [round(float(x) * 1000, 1) for x in d], "z max", [round(float(x) * 1000, 1) for x in S1[:, :, 2].max(1).values], flush=True)
