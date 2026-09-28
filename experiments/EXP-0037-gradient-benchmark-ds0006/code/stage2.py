"""EXP-0037 stage 2 (Genesis): simulate, through ONE path (rollout_candidates,
full fidelity, training physics), every state's 128 pool pushes and every
a_rank / GD-restart endpoint from stage 1. Batches are grouped by push length
(a batch runs as long as its longest push). Outcomes are banked in DS-0004
(sim_path oracle_rollout_full; fingerprint carries the physics), so a cut-off
run resumes from the bank; the per-state index is checkpointed after every state.
"""
import hashlib, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.gt_bank import GroundTruthBank, state_key
from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics, project_push
from simple_mpc.oracle_mpc import load_oracle_config
from utils import git_provenance

ART = REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/artifacts/RUN-0001"
CFG = REPO / "simple_mpc/config/config_oracle.yaml"
K = 32


def main():
    s1 = torch.load(ART / "stage1.pt", weights_only=False)
    out_p = ART / "stage2_index.pt"
    out = torch.load(out_p, weights_only=False) if out_p.exists() else {
        "labels": {}, "state_keys": {}, "provenance": git_provenance()}
    cfg = oracle_config_with_physics(load_oracle_config(str(CFG)))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    fp = {"config_sha1": hashlib.sha1(CFG.read_bytes()).hexdigest()[:12], "use_rollout_fidelity": False,
          "n_envs": K, "physics": dict(TRAINING_PHYSICS), "n_particles": 20}
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env)
    import genesis as gs
    bank = GroundTruthBank(REPO / "datasets/DS-0004-ground-truth-bank/data")
    for s in s1["states"]:
        if s in out["labels"]:
            continue
        t0 = time.time()
        st = s1["states0"][s].float()
        snap = {"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
        acts, labels = [], []
        for j, a in enumerate(s1["pool_actions"][s]):
            acts.append(a); labels.append(("pool", j, -1, -1))
        for arm, rec in s1["arms"].items():
            for g in range(len(s1["goals"])):
                acts.append(rec["a_rank"][s][g]); labels.append(("rank", arm, g, -1))
                for r in range(rec["a_restart"][s].shape[1]):
                    acts.append(rec["a_restart"][s][g, r]); labels.append(("restart", arm, g, r))
        A = project_push(torch.stack(acts).float().cpu())[0]
        uniq = torch.unique(A, dim=0)

        def simulate(batch):
            order = (batch[:, 2:] - batch[:, :2]).norm(dim=1).argsort()
            fin = torch.zeros(len(batch), st.shape[0], 3, device="cpu")
            for i in range(0, len(batch), K):
                o = order[i:i + K]; n = len(o)
                chunk = batch[o]
                pad = torch.cat([chunk, chunk[:1].expand(K - n, -1)]) if n < K else chunk
                pos = env.rollout_candidates(pad[:, None, :], snap, use_rollout_fidelity=False, record=False)
                fin[o] = pos.float().cpu()[:n]
            return fin
        hit, _ = bank.lookup(st, uniq, "oracle_rollout_full")
        bank.evaluate(st, uniq, "oracle_rollout_full", fp, "EXP-0037/RUN-0002", simulate)
        out["labels"][s] = {"labels": labels, "actions": A}
        out["state_keys"][s] = state_key(st)
        tmp = Path(str(out_p) + ".tmp"); torch.save(out, tmp); tmp.replace(out_p)
        print(f"state {s:2d}: {len(A)} pushes, {len(uniq)} unique, {int((~hit).sum())} simulated "
              f"({time.time() - t0:.0f}s)", flush=True)
    env.destroy()
    print("wrote", out_p)


if __name__ == "__main__":
    main()
