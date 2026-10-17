import sys, time, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim, legalize
from transforms.functional import action_to_pose
sid = sys.argv[1]; E = int(sys.argv[2])
s = [x for x in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"] if x["id"] == sid][0]
sim = Sim(s["n"], E); st = torch.tensor(s["state0"], device="cpu").float()[None].expand(E, -1, -1).contiguous()
bank = sim.bank(st, 64); a, _, ok = legalize(bank[0].cuda(), st[0]); A = a[ok][:E].cpu()
print("push lengths mm", [round(float((x[2:] - x[:2]).norm()) * 1000) for x in A], flush=True)
sim.set(st); gs = sim.gs; sx, sy, ex, ey, ang = action_to_pose(A.to(gs.device)); z = torch.full_like(sx, float(sim.sim._operation_height))
cnt = [0]; t0 = time.time()
def on_step(*args, **kw):
    cnt[0] += 1
    if cnt[0] % 25 == 0: print(f"  step {cnt[0]} t {time.time() - t0:.1f}s", flush=True)
try:
    sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang, on_step=on_step)
except TypeError:
    sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
t1 = time.time(); sim.sim.update_material_state(); t2 = time.time()
print(f"{sid} E{E}: execute {t1 - t0:.1f}s ({cnt[0]} steps) settle {t2 - t1:.1f}s", flush=True)
