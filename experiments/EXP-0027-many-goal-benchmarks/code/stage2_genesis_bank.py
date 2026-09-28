"""EXP-0027 item 2, stage 2 (Genesis): simulate, through ONE execution path,
every action stage 1 produced plus each state's 100-action pool, and bank the
final particle states (DS-0004, sim_path "oracle_rollout_full").

Path: GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=False) -- the
same full-fidelity path EXP-0023 stage 2 used for its final batch, restoring
the slate state as a frozen snapshot exactly as it did. No CEM oracle.

Also simulates EXP-0023's own a_rank / a_grad for slates 0-9 (corner), which
(a) checks this pipeline against EXP-0023's recorded stage-2 dv and (b) gives
a second, numerically-perturbed GD run of the same (model, state, goal) --
a direct measurement of optimiser noise in TRUE dv.

Outputs only final states (via the bank) + the action index; dv under any
goal is computed in stage 3.
"""
from __future__ import annotations
import argparse, glob, hashlib, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))

from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.gt_bank import GroundTruthBank, state_key
from simple_mpc.oracle_mpc import load_oracle_config
from stage1_optimise import project
from utils import git_provenance

BANK = REPO / "datasets/DS-0004-ground-truth-bank/data"
CFG = REPO / "simple_mpc/config/config_oracle.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--slates", type=int, nargs="*", default=None)
    args = ap.parse_args()
    s1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    ref = torch.load(glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage1_actions.pt"))[0],
                     map_location="cpu", weights_only=False)
    slates = args.slates if args.slates is not None else s1["slates"]
    K = args.n_envs

    cfg = load_oracle_config(str(CFG))
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = K
    fingerprint = {"config_sha1": hashlib.sha1(CFG.read_bytes()).hexdigest()[:12],
                   "use_rollout_fidelity": False, "n_envs": K, "n_particles": 20}
    env = GenesisOracleEnv(cfg, n_envs=K)
    import genesis as gs
    bank = GroundTruthBank(BANK)
    out = {"slates": slates, "labels": {}, "state_keys": {}, "restore_err_m": {},
           "fingerprint": fingerprint, "provenance": git_provenance()}

    for s in slates:
        t0 = time.time()
        st = s1["states0"][s].float()
        snap = {"pos": st[None, :, 0:3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
        env.restore_snapshot(snap)
        back = env.current_particles_world().float().cpu()
        out["restore_err_m"][s] = float((back[0] - st[:, 0:3]).abs().max())

        acts, labels = [], []
        for j, a in enumerate(s1["pool_actions"][s]):
            acts.append(a); labels.append(("pool", j, None, None))
        for arm, rec in s1["arms"].items():
            for gi in range(len(s1["goals"])):
                acts.append(rec["a_rank"][s][gi]); labels.append(("rank", arm, gi, None))
                acts.append(rec["a_grad"][s][gi]); labels.append(("grad", arm, gi, None))
        if s in ref["slates"]:
            for arm, rec in ref["arms"].items():
                acts.append(rec["a_rank"][s]); labels.append(("rank23", arm, 0, None))
                acts.append(rec["a_grad"][s]); labels.append(("grad23", arm, 0, None))
        A = torch.stack(acts).float()
        A = project(A)[0]                       # the legal action that actually runs
        # dedupe exact rows; simulate only what the bank does not already hold
        uniq, inv = torch.unique(A, dim=0, return_inverse=True)

        def simulate(batch):
            finals = []
            for i in range(0, len(batch), K):
                chunk = batch[i:i + K]
                n = len(chunk)
                pad = torch.cat([chunk, chunk[:1].expand(K - n, -1)]) if n < K else chunk
                pos = env.rollout_candidates(pad[:, None, :], snap, use_rollout_fidelity=False,
                                             record=False)
                finals.append(pos.float().cpu()[:n])   # (n, N, 3): positions only
            return torch.cat(finals)

        hit0, _ = bank.lookup(st, uniq, "oracle_rollout_full")
        bank.evaluate(st, uniq, "oracle_rollout_full", fingerprint, "EXP-0027/RUN-0003", simulate)
        out["labels"][s] = {"labels": labels, "actions": A, "uniq": uniq, "inv": inv}
        out["state_keys"][s] = state_key(st)
        print(f"slate {s:2d}: {len(A)} actions, {len(uniq)} unique, {int((~hit0).sum())} simulated, "
              f"restore_err {out['restore_err_m'][s]:.1e} ({time.time() - t0:.1f}s)", flush=True)
        torch.save(out, args.out)
    env.destroy()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
