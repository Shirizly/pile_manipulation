"""EXP-0049: which candidate sampler gives useful 20 mm perpendicular pushes
on n=20 single-layer states (scatter + constructed clumps)?

Phases (each checkpointed atomically, a cut-off run resumes):
  clumps : build 10 single-layer clump states with `make_clump_states`
           (reusable), save artifacts/clump_states.pt
  sim    : for every (state, sampler) draw 96 candidates from
           SandboxManipulation.generate_action_samples, record the raw
           geometry (violations), convert to a 4-D [sx,sy,ex,ey] action with
           start + direction kept and length forced to exactly 20 mm (4-D =>
           yaw derived perpendicular to travel), simulate with
           rollout_candidates (full fidelity, training physics), save
           artifacts/sim/<state>_<sampler>.pt
  analyse: analyse.py (CPU only) -> results/*.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
EXP = REPO / "experiments/EXP-0049-sampler-study"
ART = EXP / "artifacts"
CFG = REPO / "simple_mpc/config/config_oracle.yaml"
K = 32
N_CAND = 96
L = 0.020
CUBE = 0.005
Z0 = 0.012471          # settled layer-0 cube-centre height (DS-0006 states)
CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys"

SAMPLERS = {
    "S1_pile": dict(pile_aware=True, min_swath_particles=3, push_length=L),
    "S2_pile_relaxed": dict(pile_aware=True, min_swath_particles=1,
                            pile_clearance=0.5 * CUBE, push_length=L),
    "S3_placement_perp": dict(placement_aware=True, perpendicular_pushes=True, push_length=L),
    "S4_blind_perp": dict(placement_aware=False, perpendicular_pushes=True, push_length=L),
}


def save_atomic(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if str(path).endswith(".json"):
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- clump states
def propose_clump_layouts(n_layouts: int, n_particles: int = 20, gen=None,
                          spacing: float = CUBE + 0.0005, tray_half: float = 0.064,
                          margin: float = 0.008, max_clusters: int = 3):
    """CPU proposal of `n_layouts` single-layer clump layouts, each (P,7)
    [x,y,z,qw,qx,qy,qz]: the P cubes split into 1..max_clusters tight lattice
    patches (pitch `spacing`), each with a random centre and yaw inside the
    tray, cubes aligned with their patch. Patches never overlap (min
    inter-cube distance >= 0.95*spacing). Unsettled -- see make_clump_states."""
    with torch.device("cpu"):      # Genesis makes cuda the default device
        return _propose(n_layouts, n_particles, gen or torch.Generator().manual_seed(0),
                        spacing, tray_half, margin, max_clusters)


def _propose(n_layouts, n_particles, gen, spacing, tray_half, margin, max_clusters):
    out = []
    while len(out) < n_layouts:
        k = int(torch.randint(1, max_clusters + 1, (1,), generator=gen))
        # split P into k sizes >= 4
        cuts = sorted(torch.randperm(n_particles - 4 * k + k - 1, generator=gen)[:k - 1].tolist())
        sizes, prev = [], -1
        for c in cuts + [n_particles - 4 * k + k - 1]:
            sizes.append(c - prev - 1 + 4); prev = c
        pts, yaws = [], []
        ok = True
        for m in sizes:
            cols = int(math.ceil(math.sqrt(m) * (1 + 0.6 * float(torch.rand(1, generator=gen)))))
            cols = max(2, min(cols, m))
            ij = torch.tensor([(i // cols, i % cols) for i in range(m)], dtype=torch.float32)
            local = (ij - ij.mean(0)) * spacing
            th = float(torch.rand(1, generator=gen)) * math.pi
            R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            local = local @ R.T
            ext = local.abs().max(0).values + CUBE * 0.75
            lim = tray_half - margin - ext
            if (lim <= 0).any():
                ok = False; break
            c = (torch.rand(2, generator=gen) * 2 - 1) * lim
            pts.append(local + c); yaws += [th] * m
        if not ok:
            continue
        xy = torch.cat(pts)
        d = torch.cdist(xy, xy) + torch.eye(n_particles)
        if d.min() < 0.95 * spacing:
            continue
        yaw = torch.tensor(yaws)
        q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                         torch.sin(yaw / 2)], 1)
        z = torch.full((n_particles, 1), Z0 + 0.0003)
        out.append(torch.cat([xy, z, q], 1))
    return torch.stack(out)


def make_clump_states(sim, n_states: int, seed: int = 0, n_particles: int = 20,
                      z_tol: float = 0.5 * CUBE, max_rounds: int = 10):
    """Reusable generator. `sim` is a built SandboxManipulation (e.g.
    GenesisOracleEnv._sim) with training physics applied. Proposes one layout
    per env, writes them with set_particle_state, settles with
    update_material_state, and keeps states whose every cube stays in layer 0
    (|z - Z0| < z_tol) and inside the tray. Returns (n_states, P, 7) CPU
    settled states and a stats dict."""
    import genesis as gs
    gen = torch.Generator(device="cpu").manual_seed(seed)
    E = sim._n_envs
    kept, stats = [], {"proposed": 0, "rejected_z": 0, "rejected_tray": 0}
    for _ in range(max_rounds):
        lay = propose_clump_layouts(E, n_particles, gen)
        sim.set_particle_state(lay[:, :, :3].to(gs.device).contiguous(),
                               lay[:, :, 3:7].to(gs.device).contiguous())
        sim.update_material_state()
        st = sim._particle_state[:, :, :7].float().cpu()
        stats["proposed"] += E
        for e in range(E):
            z_ok = bool(((st[e, :, 2] - Z0).abs() < z_tol).all())
            t_ok = bool((st[e, :, :2].abs() < 0.064).all())
            if not z_ok:
                stats["rejected_z"] += 1
            elif not t_ok:
                stats["rejected_tray"] += 1
            else:
                kept.append(st[e])
        if len(kept) >= n_states:
            break
    return torch.stack(kept[:n_states]), stats


# ---------------------------------------------------------------- simulation
def geometry_report(starts, stops, angles):
    """Raw sampler output (S,3),(S,3),(S,) -> per-candidate length and
    perpendicularity error (rad, in [0, pi/2]) vs the 4-D convention
    yaw = atan2(d) + pi/2 (plate pi-periodic)."""
    d = (stops - starts)[:, :2]
    length = d.norm(dim=1)
    yaw4 = torch.atan2(d[:, 1], d[:, 0]) + math.pi / 2
    e = torch.remainder(yaw4 - angles + math.pi / 2, math.pi) - math.pi / 2
    return length, e.abs()


def to_fixed_action(starts, stops, angles):
    """Keep start and travel direction; force length to exactly L. Direction
    falls back to the blade normal (angle - pi/2) if the raw push is ~0."""
    d = (stops - starts)[:, :2]
    n = d.norm(dim=1, keepdim=True)
    normal = torch.stack([torch.cos(angles - math.pi / 2), torch.sin(angles - math.pi / 2)], 1)
    u = torch.where(n > 1e-6, d / n.clamp_min(1e-9), normal)
    s = starts[:, :2]
    return torch.cat([s, s + L * u], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs states i, i+n, ...")
    args = ap.parse_args()
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    from Genesis.binned_slate_dataset import BinnedSlateCorpus
    from utils import git_provenance
    import genesis as gs

    cfg = oracle_config_with_physics(load_oracle_config(str(CFG)))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env)
    sim = env._sim

    # ---- states
    clump_p = ART / "clump_states.pt"
    if clump_p.exists():
        clumps = torch.load(clump_p)["states"]
    else:
        t0 = time.time()
        clumps, cstats = make_clump_states(sim, 10, seed=0)
        save_atomic({"states": clumps, "stats": cstats, "physics": dict(TRAINING_PHYSICS),
                     "generator": "experiments/EXP-0049-sampler-study/code/sampler_study.py::make_clump_states(seed=0)",
                     "provenance": git_provenance()}, clump_p)
        print(f"clumps: {cstats} ({time.time()-t0:.0f}s)", flush=True)
    t = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    scatter = [t.states[(t.slate_idx == s).nonzero()[0, 0]].float() for s in range(120, 130)]
    states = {f"scatter{j}": scatter[j] for j in range(10)}
    states.update({f"clump{j}": clumps[j] for j in range(10)})
    order = [f"{kind}{j}" for j in range(10) for kind in ("scatter", "clump")]
    if args.pilot:
        order = order[:1]
    si, sn = map(int, args.shard.split("/"))
    order = order[si::sn]

    n_per_env = N_CAND // K
    for name in order:
        st = states[name]
        snap = {"pos": st[None, :, :3].to(gs.device).contiguous(),
                "quat": st[None, :, 3:7].to(gs.device).contiguous()}
        for sname, kw in SAMPLERS.items():
            out_p = ART / "sim" / f"{name}_{sname}.pt"
            if out_p.exists():
                continue
            env.restore_snapshot(snap)
            torch.manual_seed(hash((name, sname)) % 2**31)
            t0 = time.time()
            s_, e_, a_ = sim.generate_action_samples(n_per_env, **kw)
            t_sample = time.time() - t0
            starts = s_.reshape(-1, 3).float().cpu(); stops = e_.reshape(-1, 3).float().cpu()
            angles = a_.reshape(-1).float().cpu()
            length, perp_err = geometry_report(starts, stops, angles)
            A = to_fixed_action(starts, stops, angles)
            t0 = time.time()
            fin = torch.zeros(N_CAND, st.shape[0], 3, device="cpu")
            for i in range(0, N_CAND, K):
                chunk = A[i:i + K]
                pos = env.rollout_candidates(chunk[:, None, :], snap, use_rollout_fidelity=False, record=False)
                fin[i:i + K] = pos.float().cpu()
            t_sim = time.time() - t0
            fl = (A[:, 2:] - A[:, :2]).norm(dim=1)   # 4-D => perpendicular by construction
            save_atomic({"state": st, "raw_starts": starts, "raw_stops": stops, "raw_angles": angles,
                         "raw_length": length, "raw_perp_err": perp_err, "actions": A,
                         "final_len": fl, "final": fin, "t_sample": t_sample, "t_sim": t_sim,
                         "sampler_kw": kw}, out_p)
            print(f"{name} {sname}: sample {t_sample:.2f}s sim {t_sim:.0f}s | raw len "
                  f"[{length.min()*1e3:.1f},{length.max()*1e3:.1f}]mm perp_err max {perp_err.max():.3g} | "
                  f"moved>1mm mean {((fin[:, :, :2]-st[None, :, :2]).norm(dim=-1) > 1e-3).sum(1).float().mean():.2f}",
                  flush=True)
    env.destroy()


if __name__ == "__main__":
    main()
