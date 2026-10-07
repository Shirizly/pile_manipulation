"""EXP-0075 diagnosis D1: audit + unit tests of the multi-push path (no simulator).
T1  WideEns.predict (the planner's model path) == the validated eval path (eval_wide Zoom/World -> Ens -> posthoc balance) for one push, and the first output of an H=2 / H=4 call == the H=1 output.
T2  Inside plan_seq on real recorded states: (a) how many pool sequences have a later push that predictably does nothing, (b) spread of the elites' later pushes, (c) does the shifted warm-start plan enter the pool and where does it rank,
    (d) what the final best sequence looks like per step (predicted change of the state), (e) cost found at H=1 vs the same first push inside the best H=2 / H=4 sequence.
usage: PYTHONPATH=. python -u diag_unit.py"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0075-closed-loop-benchmark/code"); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import wide_planner as wp
import eval_wide as ew
import eval_variants as ev
from posthoc_fix import fix_delta
from batched_closed_loop import goal_mask, goal_dist
from simple_mpc.learned_mpc import project_push
R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"
torch.manual_seed(0); np.random.seed(0)
model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
D = json.load(open("experiments/EXP-0075-closed-loop-benchmark/results/main_Lfull.json"))
eps = [e for e in D["episodes"] if e["cell"] == "cem4_Lfull"]
e0 = [e for e in eps if e["goal"] == "letter_T_w20" and e["start"] == 40][0]
st = lambda e, t: torch.cat([torch.tensor(e["states"][t]), torch.tensor(e["quats"][t])], 1).float()
S0 = st(e0, 3)
dist, mask = goal_dist("letter_T_w20"), torch.from_numpy(goal_mask("letter_T_w20"))


def rand_push(n):
    c = S0[torch.randint(0, len(S0), (n,)), :2]; ang = torch.rand(n) * 2 * np.pi; d = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    L = 0.02 + 0.05 * torch.rand(n, 1); s = c - d * 0.03; return torch.cat([s, s + d * L], 1)


# ---- T1
model.begin(S0); N = 8; A = torch.stack([rand_push(N) for _ in range(4)], 1)       # (N, 4, 4) random 4-push sequences
A, _ = project_push(A.reshape(-1, 4), 0.02, 0.07)[0], None; A = A.reshape(N, 4, 4).to(wp.DEV)
p4 = model.predict(A[..., :2], A[..., 2:]); p1 = model.predict(A[:, :1, :2], A[:, :1, 2:]); p2 = model.predict(A[:, :2, :2], A[:, :2, 2:])
print("T1a first output of H=4 call vs H=1 call, max abs diff:", float((p4[0] - p1[0]).abs().max()), " H=2 vs H=1:", float((p2[0] - p1[0]).abs().max()), " H=2 vs H=4 (step 2):", float((p2[1] - p4[1]).abs().max()))
ens = ev.Ens([ew.Zoom(model.members[0].net, 128), ew.World(model.members[1].net, 128)])
S0b = S0[None].expand(N, -1, -1).contiguous(); raw = ens.rollout(S0b, A[:, :1, :2].cpu(), A[:, :1, 2:].cpu(), None)[0]
o0 = ew.occ_from_particles(S0[None]).expand(N, -1, -1); ref = (o0 + fix_delta(raw - o0, 0., "balance")).clamp(0, 1)
print("T1b planner path vs validated eval path (Ens + balance), max abs diff:", float((p1[0].cpu() - ref).abs().max()))

# ---- T2
obj = wp.SeqObjective(model, S0, dist, mask, 1.0)
# bank stand-in: pile-aware-like pushes (start 25-45 mm before a random cube, aimed at it) -- good enough to look at the mechanics
def bank_like(m):
    c = S0[torch.randint(0, len(S0), (m,)), :2]; ang = torch.rand(m) * 2 * np.pi; d = torch.stack([torch.cos(ang), torch.sin(ang)], 1); L = 0.02 + 0.05 * torch.rand(m, 1)
    s = c - d * (0.01 + 0.02 * torch.rand(m, 1)); return torch.cat([s, s + d * L], 1)
bank = bank_like(4096)
for H in (1, 2, 4):
    torch.manual_seed(1); t0 = time.time(); out = wp.plan_seq(obj, bank, "cem", H); seq = out["seq"].to(wp.DEV)
    preds = model.predict(seq[None, :, :2], seq[None, :, 2:]); occ0 = model.occ0
    dm = [float((preds[0] - occ0).abs().sum())] + [float((preds[k] - preds[k - 1]).abs().sum()) for k in range(1, H)]
    print(f"T2 H={H}: best cost {out['cost']:+.4f} ({time.time()-t0:.1f}s); predicted |change of the occupancy| per push (pixels of mass): {np.round(dm, 2)}; first push length {float((seq[0,2:]-seq[0,:2]).norm())*1000:.0f} mm")
# warm start: plan H=4, then plan again at the SAME state with warm=plan[1:]
torch.manual_seed(2); out4 = wp.plan_seq(obj, bank, "cem", 4); warm = out4["seq"][1:]
H = 4; n_pool = 256; first = bank[:n_pool].to(wp.DEV); lo, hi = 0.02, 0.07; proj = lambda x: project_push(x, lo, hi)[0]
b = proj(bank.to(wp.DEV).float()); later = b[torch.randint(0, len(b), (n_pool, H - 1))]; pool = torch.cat([proj(first)[:, None], later], 1)
c_nowarm = obj.cost(pool); w = warm.to(wp.DEV).float(); nw = 16
pool_w = pool.clone(); pool_w[:nw, : w.shape[0]] = proj((w[None] + 0.005 * torch.randn(nw, w.shape[0], 4, device=wp.DEV)).reshape(-1, 4)).reshape(nw, -1, 4); c_w = obj.cost(pool_w)
print("T2 warm-start rows (index 0-15) cost rank in the 256-row pool (0 = best):", c_w.argsort().argsort()[:16].tolist(), "| cost of warm rows mean", float(c_w[:16].mean()), "vs pool median", float(c_w.median()), "| warm rows have steps 2-3 from the previous plan and a RANDOM last step (index 3)")
el = pool_w[c_w.argsort()[:32]]; print("T2 elite spread (std, mm) per step for the initial pool (H=4):", [round(float(el[:, k].std(0).mean()) * 1000, 1) for k in range(4)], "(first step is pile-aware; later steps are random bank rows)")
print("T2 fraction of pool sequences whose step-2 push predictably changes < 0.5 px of mass:", float(np.mean([float((p - q).abs().sum()) < 0.5 for p, q in zip(model.predict(pool[:, :2, :2], pool[:, :2, 2:])[1][:128], model.predict(pool[:, :1, :2], pool[:, :1, 2:])[0][:128])])))
