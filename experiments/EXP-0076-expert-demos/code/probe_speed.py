"""Cost pilot: seconds per batched push (execute vs settle) vs env count E, n20 / n50. No outcomes."""
import sys, time, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim, legalize
n = int(sys.argv[1]); Es = [int(x) for x in sys.argv[2:]]
sc = [s for s in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"] if s["n"] == n][:8]
res = {}
for E in Es:
    sim = Sim(n, E); st = torch.stack([torch.tensor(sc[i % len(sc)]["state0"]).float() for i in range(E)])
    bank = sim.bank(st, 64)
    for rep in range(3):
        acts = []
        for i in range(E):
            a, _, ok = legalize(bank[i].cuda(), st[i]); acts.append(a[ok][rep].cpu())
        sim.set(st); t0 = time.time()
        from transforms.functional import action_to_pose
        gs = sim.gs; sx, sy, ex, ey, ang = action_to_pose(torch.stack(acts).to(gs.device)); z = torch.full_like(sx, float(sim.sim._operation_height))
        sim.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); t1 = time.time(); sim.sim.update_material_state(); t2 = time.time()
        print(f"n{n} E{E} rep{rep}: execute {t1 - t0:.2f}s settle {t2 - t1:.2f}s -> {E / (t2 - t0):.2f} sims/s", flush=True)
        res.setdefault(E, []).append((t1 - t0, t2 - t1))
    t0 = time.time(); sim.bank(st, 4096); print(f"  bank 4096/env: {time.time() - t0:.2f}s", flush=True)
    sim.destroy()
print(json.dumps({E: v for E, v in res.items()}))
