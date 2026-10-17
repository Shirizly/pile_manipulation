"""EXP-0068 / DS-0023 collection: same-state candidate pools from VISITED closed-loop states (GPU, Genesis).

For every state in starts.pt (select_states.py): broadcast it to all envs (set_particle_state + update_material_state,
as Genesis/chain_collection.py --mode pools), draw --pool candidates with DS-0016's sampler (exact 20 mm, perpendicular,
redraw until valid), add the planner's recorded top-k candidates passed through Genesis.action_sampling.legalize_pushes
(the closed loop's own pre-execution call; un-legalisable ones kept and flagged legal_ok False), simulate every candidate
from the identical settled state in sub-batches of --n-envs, and write one atomic pools_<i>.pt per state (resumable),
then merge into data/pools_0.pt (unified benchmark shape + source / legality columns + per-pool meta).
Usage: python -u collect_pools.py [--starts ...] [--out ...] [--pool 64] [--n-envs 32] [--limit N]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Genesis.chain_collection import check_actions  # noqa: E402

SAMPLER = {"pile_aware": True, "min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}
DATA = REPO / "datasets/DS-0023-visited-state-pools/data"


def _atomic_save(obj, path: Path):
    tmp = Path(str(path) + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", default=str(DATA / "starts.pt"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--push-length", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=68)
    ap.add_argument("--limit", type=int, default=None, help="only the first N states (cost pilot)")
    ap.add_argument("--no-planner", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    S = torch.load(a.starts, map_location="cpu", weights_only=False)
    starts, topk, topk_dv, meta = S["states"].float(), S["topk_actions"].float(), S["topk_dv"].float(), S["meta"]
    n_states = len(starts) if a.limit is None else min(a.limit, len(starts))

    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, XY_MARGIN, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    from simple_mpc.adapters import OCC_BOUNDS
    from Genesis.action_sampling import legalize_pushes
    from model.retrieval.frame import yaw_from_quat
    from transforms.functional import action_to_pose
    physics = dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")), physics)
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = a.n_envs
    env = GenesisOracleEnv(cfg, n_envs=a.n_envs); apply_physics(env, physics); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    E = a.n_envs

    def set_state(one):                        # (1, P, 7) broadcast to every env; returns the settled state (E, P, 7)
        sim.set_particle_state(one[:, :, :3].to(gs.device), one[:, :, 3:7].to(gs.device))
        sim.update_material_state()
        return sim._particle_state.detach().cpu().float().clone()

    def push(pS, pE, ang):
        sim.execute_action(pS.to(gs.device), pE.to(gs.device), ang.to(gs.device))
        sim.update_material_state()
        return sim._particle_state.detach().cpu().float().clone()

    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else dict(
        physics=physics, sampler=SAMPLER, push_length=a.push_length, pool=a.pool, n_envs=a.n_envs, seed=a.seed,
        starts=str(a.starts), states={}, complete=False, argv=sys.argv)
    for i in range(n_states):
        f = out / f"pools_state{i:03d}.pt"
        if f.exists():
            continue
        t0 = time.time()
        one = starts[i:i + 1]
        settled = set_state(one)                                  # every env identical
        drift = float((settled[:, :, :2] - one[:, :, :2]).norm(dim=-1).max())
        one_s = settled[:1].clone()                               # the pool's start state = the settled broadcast
        # --- sampler candidates (DS-0016 recipe): keep only exact-length perpendicular draws
        acts, need, tries = [], a.pool, 0
        while need > 0 and tries < 50:
            ps, pe, an = sim.generate_action_samples(1, **SAMPLER)
            ps, pe, an = ps[:, 0].float().cpu(), pe[:, 0].float().cpu(), an[:, 0].float().cpu()
            g = getattr(sim, "_last_gap_out_of_window", None)
            gow = g[:, 0].clone() if g is not None else torch.zeros(E, dtype=torch.bool, device="cpu")
            ok = check_actions(ps, pe, an, a.push_length)[0]
            acts.append((ps[ok], pe[ok], an[ok], gow[ok])); need -= int(ok.sum()); tries += 1
        pS = torch.cat([x[0] for x in acts])[:a.pool]; pE = torch.cat([x[1] for x in acts])[:a.pool]
        AN = torch.cat([x[2] for x in acts])[:a.pool]; GOW = torch.cat([x[3] for x in acts])[:a.pool]
        n_s = len(pS)
        src = torch.zeros(n_s, dtype=torch.long, device="cpu"); lok = torch.ones(n_s, dtype=torch.bool, device="cpu")
        lsh = torch.zeros(n_s, device="cpu"); pdv = torch.full((n_s,), float("nan"), device="cpu")
        # --- planner candidates: legalised exactly as the closed loop legalises its chosen push
        if not a.no_planner:
            tk = topk[i]; K = len(tk)
            st = one_s.expand(K, -1, -1)
            a2, s2, ok2 = legalize_pushes(tk, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001,
                                          box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))
            sx, sy, ex, ey, ang = action_to_pose(a2.float())
            z = torch.full_like(sx, float(sim._operation_height))
            pS2 = torch.stack([sx, sy, z], 1); pE2 = torch.stack([ex, ey, z], 1)
            pS = torch.cat([pS, pS2]); pE = torch.cat([pE, pE2]); AN = torch.cat([AN, ang.float()])
            GOW = torch.cat([GOW, torch.zeros(K, dtype=torch.bool, device="cpu")])
            src = torch.cat([src, torch.ones(K, dtype=torch.long, device="cpu")]); lok = torch.cat([lok, ok2.cpu()])
            lsh = torch.cat([lsh, s2.cpu()]); pdv = torch.cat([pdv, topk_dv[i]])
        valid = check_actions(pS, pE, AN, a.push_length)[0]
        n = len(pS); outs = []
        for j in range(0, n, E):
            m = min(E, n - j)
            pad = lambda x: torch.cat([x[j:j + m], x[j:j + 1].expand(E - m, *x.shape[1:])]) if m < E else x[j:j + m]
            set_state(one_s)
            outs.append(push(pad(pS), pad(pE), pad(AN))[:m])
        states_ = torch.cat(outs)
        rec = dict(states=one_s.expand(n, -1, -1).clone(), states_=states_, p_starts=pS, p_stops=pE, angles=AN,
                   pool_idx=torch.full((n,), i, dtype=torch.long, device="cpu"), valid=valid, gap_out_of_window=GOW,
                   source=src, legal_ok=lok, legal_shift_m=lsh, planner_pred_dv=pdv, meta=meta[i],
                   settle_drift_m=drift, given_state=one[0].clone())
        _atomic_save(rec, f)
        disp = (states_[:, :, :2] - one_s[:, :, :2]).norm(dim=-1).max(1).values
        man["states"][str(i)] = dict(rows=n, n_sampler=n_s, valid=int(valid.sum()), gap_out_of_window=int(GOW.sum()),
                                     planner_legal_ok=int(lok[src == 1].sum()) if n > n_s else 0,
                                     null_rows=int((disp < 0.001).sum()), settle_drift_m=drift, seconds=time.time() - t0,
                                     model=meta[i]["model"], goal=meta[i]["goal"], k=meta[i]["k"])
        tmp = Path(str(man_p) + ".tmp"); tmp.write_text(json.dumps(man, indent=1)); os.replace(tmp, man_p)
        print(f"state {i}: {n} rows ({n_s} sampler), valid {int(valid.sum())}, planner legal {man['states'][str(i)]['planner_legal_ok']}, "
              f"null {man['states'][str(i)]['null_rows']}, drift {drift * 1000:.2f} mm, {time.time() - t0:.0f}s", flush=True)
    # --- merge
    files = [out / f"pools_state{i:03d}.pt" for i in range(len(starts)) if (out / f"pools_state{i:03d}.pt").exists()]
    if len(files) == len(starts) or a.limit is not None:
        recs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        keys = ["states", "states_", "p_starts", "p_stops", "angles", "pool_idx", "valid", "gap_out_of_window", "source",
                "legal_ok", "legal_shift_m", "planner_pred_dv"]
        data = {k: torch.cat([r[k] for r in recs]) for k in keys}
        data["meta"] = [r["meta"] for r in recs]; data["start_kind"] = ["visited"] * len(data["states"])
        data["settle_drift_m"] = [r["settle_drift_m"] for r in recs]
        _atomic_save(data, out / "pools_0.pt")
        man["complete"] = len(files) == len(starts)
        tmp = Path(str(man_p) + ".tmp"); tmp.write_text(json.dumps(man, indent=1)); os.replace(tmp, man_p)
        print("merged", len(files), "states ->", out / "pools_0.pt", "rows", len(data["states"]))
    env.destroy()


if __name__ == "__main__":
    main()
