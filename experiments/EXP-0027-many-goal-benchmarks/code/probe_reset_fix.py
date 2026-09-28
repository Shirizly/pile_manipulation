"""Does a full scene.reset() before every rollout_candidates call remove the
history dependence found by probe_batch_dependence.py? Same slate/actions.
  A1 (fresh) ; B (pool companions) ; reset ; A1 again ; reset ; C (32 x action 0)
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
s = 0; st = r["states0"][s].float()
test = project(torch.stack([r["arms"][a][k][s] for a in list(r["arms"])[:4] for k in ("a_rank", "a_grad")]))[0]
pool = project(r["pool_actions"][s].float())[0]
K = 32
cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
env = GenesisOracleEnv(cfg, n_envs=K)
import genesis as gs
snap = {"pos": st[None, :, 0:3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
v0 = lyapunov(occ_from_particles(st[None], "cpu"), dw)
scene = env._sim._scene
def run(batch, reset):
    if reset:
        scene.reset()
    pos = env.rollout_candidates(batch[:, None, :].cpu(), snap, use_rollout_fidelity=False, record=False)
    return lyapunov(occ_from_particles(pos.float().cpu(), "cpu"), dw) - v0
A = torch.cat([test, test[:1].expand(K - 8, -1)])
B = torch.cat([pool[10:34], test.flip(0)])
C = test[:1].expand(K, -1).clone()
dA1 = run(A, False); dB = run(B, False)
dA1r = run(A, True); dBr = run(B, True); dCr = run(C, True); dA1r2 = run(A, True)
dA_after_B = run(A, False)   # no reset, right after a reset+A run
print("reset+A1 vs reset+A1 again: max |ddv| %.2e" % float((dA1r - dA1r2).abs().max()))
print("reset: same 8 actions in slots 0-7 (A) vs 24-31 with pool companions (B): max |ddv| %.2e" %
      float((dA1r[:8] - dBr[24:].flip(0)).abs().max()))
print("reset+C: 32 identical actions, dv spread %.2e" % float(dCr.max() - dCr.min()))
print("reset+C vs reset+A action 0: max |ddv| %.2e" % float((dCr - dA1r[0]).abs().max()))
print("A run WITHOUT reset right after reset+A: max |ddv| vs reset+A %.2e" % float((dA_after_B - dA1r2).abs().max()))
print("first-ever batch (no reset) vs reset+A: max |ddv| %.2e" % float((dA1 - dA1r).abs().max()))
env.destroy()
