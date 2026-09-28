"""Genesis/chain_collection.py -- chained push collection from GIVEN start states.

Three modes, all writing one file per chunk (atomic, resumable: finished chunks are skipped):

  --mode chains   : K envs start from K DIFFERENT given states and each runs --steps
                  pushes in a chain (s' of push t = s of push t+1). Output `_{k}_data.pt`
                  with the training keys (states, states_, p_starts, p_stops, angles) plus
                  chain_env, chain_step, start_id, valid, single_layer, and a
                  `_{k}_config.yaml` copied from --config-template (the trainer reads
                  n_particles etc. from it), so a chains directory is directly a
                  PileSweepData training directory.
  --mode pools    : same-state pools: every start state gets --pool actions, all simulated
                  from that SAME state (one push each). Output `pools_{k}.pt` in the
                  unified benchmark shape (states, states_, p_starts, p_stops, angles,
                  pool_idx).
  --mode seqpools : same-state pools of MULTI-STEP SEQUENCES (DS-B): one chunk = one
                  shared start state, broadcast to --pool independent envs (in
                  sub-batches of --n-envs), each running its OWN --steps-push chain from
                  that identical start -- sequences diverge starting at push 0, since
                  each env's action is drawn independently by the (pile-aware) sampler
                  even though every env begins from the same state. Output `_{k}_data.pt`
                  reusing the "chains" mode's own field names/semantics
                  (chain_env = sequence id 0..pool-1, chain_step = push index 0..steps-1)
                  so it is a drop-in for `eval_retrieval.py::evaluate`'s existing
                  chain-rollout code, PLUS a `pool_idx` column (= chunk index k, constant
                  per file) mapping rows to (pool, sequence, step) as DS-B's spec requires.
                  `slates_multistep` is NOT reused here: it uses heap spawns, friction 0.3,
                  multi-layer piles and non-narrow physics (see the data-collection skill).

Start states come from --starts-file (tensor (N, n, 7)) or are generated here:
`scatter` = settled drop spawns (state_library), `clump` = a generator function given as
--clump-fn module:function (called as fn(sim, n, rng) -> (n, P, 7)).
Actions: `generate_action_samples(1, **--sampler)` per env per push; every action is
checked for exact length (--push-length, +-0.1 mm) and perpendicular yaw, and rows that
fail are kept but flagged valid=False (never silently dropped).
Physics: TRAINING_PHYSICS (simple_mpc.learned_mpc) unless --physics is given.
A manifest.json with the physics, sampler, seed and counts is rewritten after every chunk.
"""
from __future__ import annotations
import argparse, importlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FLOOR_Z_MAX = 0.015        # cube centre below this = layer 0 (floor at 0.010 m, layer-0 centre 0.0125)


def _atomic_save(obj, path: Path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


def _atomic_json(obj, path: Path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def check_actions(p_start, p_stop, angle, length, tol=1e-4, ang_tol_deg=0.1):
    """(B,3),(B,3),(B,) -> (valid (B,), realised length (B,), yaw error deg (B,)).
    Perpendicular = blade yaw equals push direction + pi/2 (mod pi), the
    transforms.functional.action_to_pose convention."""
    d = (p_stop[:, :2] - p_start[:, :2])
    L = d.norm(dim=1)
    direction = torch.atan2(d[:, 1], d[:, 0])
    err = torch.remainder(angle - direction - math.pi / 2 + math.pi / 2, math.pi) - math.pi / 2
    err_deg = err.abs() * 180 / math.pi
    ok = ((L - length).abs() <= tol) & (err_deg <= ang_tol_deg)
    return ok, L, err_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["chains", "pools", "seqpools"], default="chains")
    ap.add_argument("--n-chunks", type=int, required=True)
    ap.add_argument("--steps", type=int, default=8, help="chains/seqpools: pushes per chain")
    ap.add_argument("--pool", type=int, default=64,
                    help="pools: actions per state; seqpools: sequences per shared start state")
    ap.add_argument("--starts", choices=["scatter", "clump", "mixed", "file"], default="mixed")
    ap.add_argument("--starts-file", default=None)
    ap.add_argument("--clump-fn", default=None, help="module:function(sim, n, rng) -> (n, P, 7)")
    ap.add_argument("--sampler", default='{"placement_aware": true, "perpendicular_pushes": true, "push_length": 0.02}')
    ap.add_argument("--push-length", type=float, default=0.02)
    ap.add_argument("--physics", default=None, help="JSON physics dict (default TRAINING_PHYSICS)")
    ap.add_argument("--config-template", default="Genesis/data/overnight_randlen/scattered/cube/n20/size0.005/_0_config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    from Genesis.state_library import build_state_library
    physics = json.loads(a.physics) if a.physics else dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")), physics)
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = a.n_envs
    env = GenesisOracleEnv(cfg, n_envs=a.n_envs); apply_physics(env, physics); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    sampler = json.loads(a.sampler)
    start_gap_resolved = None
    if sampler.get("start_gap_range") is not None:
        lo_margin, hi_margin = sampler["start_gap_range"]
        start_gap_resolved = [lo_margin, a.push_length - hi_margin]  # [min, max] face-to-face gap, metres
    clump_fn = None
    if a.clump_fn:
        mod, fn = a.clump_fn.split(":")
        clump_fn = getattr(importlib.import_module(mod), fn)
    file_states = torch.load(a.starts_file, weights_only=False) if a.starts_file else None
    if isinstance(file_states, dict):
        file_states = file_states.get("states", next(iter(file_states.values())))
    template = yaml.load(open(REPO / a.config_template), Loader=yaml.UnsafeLoader) if a.mode in ("chains", "seqpools") else None
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else dict(
        mode=a.mode, physics=physics, sampler=sampler, push_length=a.push_length, starts=a.starts,
        seed=a.seed, n_envs=a.n_envs, steps=a.steps, pool=a.pool, clump_fn=a.clump_fn,
        starts_file=a.starts_file, start_gap_resolved=start_gap_resolved, chunks={}, complete=False,
        argv=sys.argv)

    def starts_for_chunk(k, count=None, force_kind=None):
        """count=None -> one state per env (a.n_envs), the "chains"/"pools" usage.
        seqpools passes count=1: ONE shared start state for the whole chunk,
        broadcast to every sequence's env by `set_particle_state`'s own (1, P, .)
        broadcast (see Genesis/sandbox_manipulation_clean.py::set_particle_state).
        `force_kind` ("scatter"/"clump") overrides `a.starts=="mixed"`'s K//2 ratio,
        which degenerates to always-scatter at count=1 -- seqpools alternates it
        per chunk instead (see the seqpools branch below) to get an even split."""
        K = a.n_envs if count is None else count
        if a.starts == "file":
            idx = [(k * K + i) % len(file_states) for i in range(K)]
            return file_states[idx].float(), ["file"] * K
        starts_mode = force_kind if force_kind is not None else a.starts
        n_clump = {"scatter": 0, "clump": K}.get(starts_mode, K // 2)
        st, kinds = [], []
        if K - n_clump:
            lib = build_state_library(sim, n_settles=1, augment=True, verbose=False)
            pick = rng.choice(len(lib), K - n_clump, replace=False)
            st.append(lib.states[pick].float().cpu()); kinds += ["scatter"] * (K - n_clump)
        if n_clump:
            st.append(clump_fn(sim, n_clump, rng).float().cpu()); kinds += ["clump"] * n_clump
        return torch.cat(st), kinds

    def _gow():
        """The ISS-010-fix sampler's own `gap_out_of_window` flag (EXP-0059,
        2026-09-28), if the last `generate_action_samples` call set one
        (only when `--sampler` carries `start_gap_range`); an all-False
        column of the right shape otherwise, so the saved schema is uniform
        regardless of sampler config."""
        g = getattr(sim, "_last_gap_out_of_window", None)
        if g is None:
            return torch.zeros(a.n_envs, dtype=torch.bool)
        return g[:, 0].clone()

    def draw_valid(max_tries: int = 20):
        """One action per env; envs whose action fails check_actions (e.g. pile-aware
        shortened it at a wall, EXP-0049) are redrawn up to max_tries times."""
        ps, pe, an = sim.generate_action_samples(1, **sampler)
        ps, pe, an = ps[:, 0].float().cpu(), pe[:, 0].float().cpu(), an[:, 0].float().cpu()
        gow = _gow()
        ok, _, _ = check_actions(ps, pe, an, a.push_length)
        for _ in range(max_tries):
            if ok.all():
                break
            p2, e2, a2 = sim.generate_action_samples(1, **sampler)
            p2, e2, a2 = p2[:, 0].float().cpu(), e2[:, 0].float().cpu(), a2[:, 0].float().cpu()
            gow2 = _gow()
            ok2, _, _ = check_actions(p2, e2, a2, a.push_length)
            take = (~ok) & ok2
            ps[take], pe[take], an[take] = p2[take], e2[take], a2[take]
            gow[take] = gow2[take]
            ok = ok | ok2
        return ps, pe, an, ok, gow

    def set_states(st):
        sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device))
        sim.update_material_state()
        return sim._particle_state.detach().cpu().float().clone()

    def push(pS, pE, ang):
        sim.execute_action(pS.to(gs.device), pE.to(gs.device), ang.to(gs.device))
        sim.update_material_state()
        return sim._particle_state.detach().cpu().float().clone()

    for k in range(a.n_chunks):
        if str(k) in man["chunks"]:
            continue
        t0 = time.time()
        if a.mode != "seqpools":
            st, kinds = starts_for_chunk(k)
            s = set_states(st)
        if a.mode == "chains":
            rec = {x: [] for x in ("states", "states_", "p_starts", "p_stops", "angles", "chain_env",
                                   "chain_step", "valid", "single_layer", "gap_out_of_window")}
            for t in range(a.steps):
                ps, pe, an, ok, gow = draw_valid()
                s_ = push(ps, pe, an)
                rec["states"].append(s); rec["states_"].append(s_); rec["p_starts"].append(ps)
                rec["p_stops"].append(pe); rec["angles"].append(an)
                rec["chain_env"].append(torch.arange(a.n_envs, device="cpu")); rec["chain_step"].append(torch.full((a.n_envs,), t, device="cpu"))
                rec["valid"].append(ok); rec["single_layer"].append((s_[:, :, 2].max(1).values < FLOOR_Z_MAX))
                rec["gap_out_of_window"].append(gow)
                s = s_
            data = {x: torch.cat(v) if x not in ("states", "states_") else torch.cat(v) for x, v in rec.items()}
            data["start_kind"] = [kinds[i] for i in data["chain_env"].tolist()]
            _atomic_save(data, out / f"_{k}_data.pt")
            conf = dict(template); conf["chain_collection"] = dict(
                physics=physics, sampler=sampler, push_length=a.push_length, steps=a.steps, chunk=k, seed=a.seed,
                start_gap_resolved=start_gap_resolved)
            conf.setdefault("box", {})["friction"] = physics.get("box_friction", conf.get("box", {}).get("friction"))
            tmp = out / f"_{k}_config.yaml.tmp"
            tmp.write_text(yaml.dump(conf)); os.replace(tmp, out / f"_{k}_config.yaml")
            n_ok = int(data["valid"].sum()); n_sl = int(data["single_layer"].sum()); n = len(data["valid"])
        elif a.mode == "pools":
            rec = {x: [] for x in ("states", "states_", "p_starts", "p_stops", "angles", "pool_idx",
                                   "valid", "gap_out_of_window")}
            snap = {"pos": None}
            for i in range(a.n_envs):
                one = s[i:i + 1]
                acts = []
                sim.set_particle_state(one[:, :, :3].to(gs.device), one[:, :, 3:7].to(gs.device))
                need = a.pool
                tries = 0
                while need > 0 and tries < 50:        # keep only valid (exact-length, perpendicular) draws
                    ps, pe, an = sim.generate_action_samples(1, **sampler)
                    ps, pe, an = ps[:, 0].float().cpu(), pe[:, 0].float().cpu(), an[:, 0].float().cpu()
                    gow = _gow()
                    ok = check_actions(ps, pe, an, a.push_length)[0]
                    acts.append((ps[ok], pe[ok], an[ok], gow[ok])); need -= int(ok.sum()); tries += 1
                ps = torch.cat([x[0] for x in acts])[:a.pool]; pe = torch.cat([x[1] for x in acts])[:a.pool]
                an = torch.cat([x[2] for x in acts])[:a.pool]; gow_all = torch.cat([x[3] for x in acts])[:a.pool]
                for j in range(0, a.pool, a.n_envs):
                    m = min(a.n_envs, a.pool - j)
                    pad = lambda x: torch.cat([x[j:j + m], x[j:j + 1].expand(a.n_envs - m, *x.shape[1:])]) if m < a.n_envs else x[j:j + m]
                    sim.set_particle_state(one[:, :, :3].to(gs.device), one[:, :, 3:7].to(gs.device))
                    s_ = push(pad(ps), pad(pe), pad(an))[:m]
                    rec["states"].append(one.expand(m, -1, -1).clone()); rec["states_"].append(s_)
                    rec["p_starts"].append(ps[j:j + m]); rec["p_stops"].append(pe[j:j + m]); rec["angles"].append(an[j:j + m])
                    rec["pool_idx"].append(torch.full((m,), k * a.n_envs + i, device="cpu"))
                    rec["valid"].append(check_actions(ps[j:j + m], pe[j:j + m], an[j:j + m], a.push_length)[0])
                    rec["gap_out_of_window"].append(gow_all[j:j + m])
            data = {x: torch.cat(v) for x, v in rec.items()}
            data["start_kind"] = kinds
            _atomic_save(data, out / f"pools_{k}.pt")
            n_ok = int(data["valid"].sum()); n = len(data["valid"]); n_sl = None
        else:  # seqpools: DS-B -- one shared start state per chunk, --pool independent
               # --steps-push sequences, sub-batched in groups of --n-envs.
            kind_choice = None
            if a.starts == "mixed":
                kind_choice = "scatter" if k % 2 == 0 else "clump"   # alternate: K//2 ratio degenerates at count=1
            elif a.starts in ("scatter", "clump"):
                kind_choice = a.starts
            st1, kinds1 = starts_for_chunk(k, count=1, force_kind=kind_choice)
            rec = {x: [] for x in ("states", "states_", "p_starts", "p_stops", "angles",
                                   "chain_env", "chain_step", "pool_idx", "valid", "single_layer",
                                   "gap_out_of_window")}
            for sub in range(0, a.pool, a.n_envs):
                m = min(a.n_envs, a.pool - sub)
                seq_ids = torch.arange(sub, sub + m, device="cpu")
                # `set_states` broadcasts the ONE shared start (st1, batch size 1)
                # to ALL a.n_envs envs (Genesis/sandbox_manipulation_clean.py::
                # set_particle_state's own (1, P, .) broadcast) -- every env in
                # this sub-batch starts identically; sequences diverge as soon as
                # each env's OWN action draw differs.
                s_seq = set_states(st1)
                for t in range(a.steps):
                    ps, pe, an, ok, gow = draw_valid()
                    s_ = push(ps, pe, an)
                    rec["states"].append(s_seq[:m]); rec["states_"].append(s_[:m])
                    rec["p_starts"].append(ps[:m]); rec["p_stops"].append(pe[:m]); rec["angles"].append(an[:m])
                    rec["chain_env"].append(seq_ids)
                    rec["chain_step"].append(torch.full((m,), t, device="cpu"))
                    rec["pool_idx"].append(torch.full((m,), k, device="cpu"))
                    rec["valid"].append(ok[:m])
                    rec["single_layer"].append((s_[:m, :, 2].max(1).values < FLOOR_Z_MAX))
                    rec["gap_out_of_window"].append(gow[:m])
                    s_seq = s_
            data = {x: torch.cat(v) for x, v in rec.items()}
            data["start_kind"] = [kinds1[0]] * len(data["valid"])
            kinds = [kinds1[0]] * a.pool   # for the manifest kind-count line below
            _atomic_save(data, out / f"_{k}_data.pt")
            conf = dict(template); conf["chain_collection"] = dict(
                physics=physics, sampler=sampler, push_length=a.push_length, steps=a.steps,
                pool=a.pool, mode="seqpools", chunk=k, seed=a.seed,
                start_gap_resolved=start_gap_resolved)
            conf.setdefault("box", {})["friction"] = physics.get("box_friction", conf.get("box", {}).get("friction"))
            tmp = out / f"_{k}_config.yaml.tmp"
            tmp.write_text(yaml.dump(conf)); os.replace(tmp, out / f"_{k}_config.yaml")
            n_ok = int(data["valid"].sum()); n_sl = int(data["single_layer"].sum()); n = len(data["valid"])
        n_gow = int(data["gap_out_of_window"].sum()) if "gap_out_of_window" in data else 0
        man["chunks"][str(k)] = dict(rows=n, valid=n_ok, single_layer_after=n_sl, seconds=time.time() - t0,
                                     gap_out_of_window=n_gow,
                                     kinds={x: kinds.count(x) for x in set(kinds)})
        _atomic_json(man, man_p)
        print(f"chunk {k}: {n} rows, {n_ok} valid, single-layer after {n_sl}, "
             f"{n_gow} gap_out_of_window, {time.time() - t0:.0f}s", flush=True)
    man["complete"] = True; _atomic_json(man, man_p)
    env.destroy()


if __name__ == "__main__":
    main()
