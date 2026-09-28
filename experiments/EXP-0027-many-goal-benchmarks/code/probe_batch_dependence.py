"""Does a push's simulated outcome depend on its env SLOT or on the OTHER
actions in the same rollout_candidates batch? (EXP-0024 only tested repeats
of 3 actions in contiguous blocks of one batch.)

Slate 0 of DS-0001, 8 test actions (EXP-0023 slate-0 a_rank/a_grad rows).
  A1: test actions in slots 0-7, slots 8-31 = copies of test action 0
  A2: A1 again (pure repeat of the identical batch)
  B : test actions REVERSED in slots 24-31, slots 0-23 = pool actions (other companions)
  C : every slot = test action 0 (32 identical envs)
Reports max |dv| differences for the same action across these placements.
"""
import glob, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))
from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.adapters import occ_from_particles, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import load_oracle_config
from stage1_optimise import project

r = torch.load(glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage1_actions.pt"))[0], weights_only=False)
s = 0
st = r["states0"][s].float()
test = torch.stack([r["arms"][a][k][s] for a in list(r["arms"])[:4] for k in ("a_rank", "a_grad")])
test = project(test)[0]
pool = project(r["pool_actions"][s].float())[0]
K = 32
cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
env = GenesisOracleEnv(cfg, n_envs=K)
import genesis as gs
snap = {"pos": st[None, :, 0:3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
v0 = lyapunov(occ_from_particles(st[None], "cpu"), dw)

def run(batch):
    pos = env.rollout_candidates(batch[:, None, :].cpu(), snap, use_rollout_fidelity=False, record=False)
    return (lyapunov(occ_from_particles(pos.float().cpu(), "cpu"), dw) - v0), pos.float().cpu()

A = torch.cat([test, test[:1].expand(K - 8, -1)])
B = torch.cat([pool[10:34], test.flip(0)])
C = test[:1].expand(K, -1).clone()
dA1, pA1 = run(A); dA2, pA2 = run(A); dB, pB = run(B); dC, pC = run(C)
print("repeat A1 vs A2 (same batch twice): max |ddv| %.2e, max |dpos| %.2e m" %
      (float((dA1 - dA2).abs().max()), float((pA1 - pA2).abs().max())))
dB8 = dB[24:].flip(0)
print("same 8 actions, slots 0-7 w/ copies vs slots 24-31 w/ pool companions: max |ddv| %.2e" %
      float((dA1[:8] - dB8).abs().max()))
print("   per action A1:", [round(float(x), 5) for x in dA1[:8]])
print("   per action B :", [round(float(x), 5) for x in dB8])
print("test action 0 in 25 slots of A1: spread %.2e ; in 32 slots of C: spread %.2e" %
      (float(torch.cat([dA1[:1], dA1[8:]]).max() - torch.cat([dA1[:1], dA1[8:]]).min()),
       float(dC.max() - dC.min())))
print("   C per-slot dv (first 8):", [round(float(x), 5) for x in dC[:8]])
env.destroy()
