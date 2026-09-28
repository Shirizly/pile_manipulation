"""C6 pipeline check: dv(corner) of EXP-0023's own a_rank/a_grad, re-simulated
through this pipeline and read back from the bank, vs EXP-0023's stage-2 dv."""
import glob, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.adapters import occ_from_particles, OCC_GRID
from simple_mpc.gt_bank import GroundTruthBank
s2 = torch.load(sys.argv[1], weights_only=False)
r2 = torch.load(glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage2_genesis.pt"))[0], weights_only=False)
bank = GroundTruthBank(REPO / "datasets/DS-0004-ground-truth-bank/data")
dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
worst = 0.0
for s in s2["slates"]:
    if s not in r2["dv"]:
        continue
    rec = bank.record("oracle_rollout_full", s2["state_keys"][s])
    L = s2["labels"][s]
    st = rec["state"]
    v0 = lyapunov(occ_from_particles(st[None], "cpu"), dw)
    for lab, a in zip(L["labels"], L["actions"]):
        if lab[0] in ("rank23", "grad23"):
            hit, f = bank.lookup(st, a[None], "oracle_rollout_full")
            dv = float(lyapunov(occ_from_particles(f[..., :3], "cpu"), dw) - v0)
            kind = lab[0][:4]
            j = r2["dv"][s]["labels"].index((kind, lab[1]))
            ref = float(r2["dv"][s]["dv_all"][j])
            worst = max(worst, abs(dv - ref))
print(f"C6: max |dv (this pipeline) - dv (EXP-0023 stage 2)| = {worst:.2e}")
