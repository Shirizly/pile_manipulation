"""PILOT (EXP-0072 active-collection plan, 2026-10-06): LEGAL large perturbations of existing n20 single-layer states.

Every op keeps the cube count (n=20, the narrow pilot's fixed scene), keeps all cubes in layer 0, and is accepted only if
(i) a MOVED cube keeps >= MIN_GAP (0.1 mm) face gap to every other cube (exact SAT, Baselines/common/cube_overlap) and the
whole final state passes SAT with squares shrunk by LEGAL_TOL (0.2 mm; the source sim states themselves touch by that much),
(ii) every cube corner is inside the tray. After the op the state
is written with sim.set_particle_state and re-settled (`settle_in_sim`), and re-checked (still layer 0, in tray, no overlap,
drift reported). Genesis-free until `settle_in_sim`.

Ops: jitter (per-cube xy+yaw, sequential rejection), relocate (k cubes to a free spot or touching another cube),
cluster_shift (rigid translation/rotation of one contact component), rigid (whole pile rotated/translated inside the tray).
`perturb_state` draws a random recipe.  CLI: python -u active_perturb.py --sanity [--sim]
"""
import math, sys, json, time
import numpy as np
sys.path.insert(0, ".")
from Baselines.common.cube_overlap import overlaps_pairs, any_overlap_matrix

CUBE = 0.005; HALF = CUBE / 2; TRAY = 0.064; Z0 = 0.012471
MIN_GAP = 0.0001; TRAY_MARGIN = 0.0003; LEGAL_TOL = 0.0002   # source states (settled sim) touch/interpenetrate by up to 0.2 mm: 'legal' = SAT with squares shrunk by LEGAL_TOL
REACH = HALF * math.sqrt(2)


def yaw_of(q):                      # (n,4) (w,x,y,z) -> yaw
    w, x, y, z = q.T; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_of(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)


def in_tray(xy, yaw, margin=TRAY_MARGIN):
    ext = HALF * (np.abs(np.cos(yaw)) + np.abs(np.sin(yaw)))      # exact axis-aligned half extent of a rotated square
    return (np.abs(xy) + ext[:, None] <= TRAY - margin).all(1)


def pair_ok(xy_a, yaw_a, xy, yaw, skip=None):
    """square a (1,2) vs all cubes (n,2): True if no overlap (with MIN_GAP). skip = index to ignore."""
    n = len(xy)
    ov = overlaps_pairs(np.repeat(xy_a, n, 0), np.repeat(yaw_a, n), xy, yaw, CUBE, tol=-MIN_GAP)
    if skip is not None: ov[skip] = False
    return not ov.any()


def legal(xy, yaw):
    return bool(in_tray(xy, yaw, 0.0).all() and not any_overlap_matrix(xy, yaw, CUBE, tol=LEGAL_TOL).any())


def split(state): return state[:, :2].astype(float).copy(), yaw_of(state[:, 3:7]).astype(float)


def pack(state, xy, yaw):
    out = state.copy(); out[:, :2] = xy; out[:, 3:7] = quat_of(yaw); out[:, 2] = Z0 + 0.0003; return out


def op_jitter(state, rng, sxy=0.004, syaw=0.5, tries=20):
    xy, yaw = split(state)
    for i in rng.permutation(len(xy)):
        for _ in range(tries):
            c = xy[i] + rng.normal(0, sxy, 2); y = yaw[i] + rng.normal(0, syaw)
            if in_tray(c[None], np.array([y]))[0] and pair_ok(c[None], np.array([y]), xy, yaw, skip=i):
                xy[i], yaw[i] = c, y; break
    return pack(state, xy, yaw)


def op_relocate(state, rng, k=3, adjacent=0.5, tries=200):
    xy, yaw = split(state)
    for i in rng.choice(len(xy), k, replace=False):
        for _ in range(tries):
            if rng.random() < adjacent:                     # touch a random OTHER cube (new contact), gap 0.3-1 mm
                j = rng.choice([q for q in range(len(xy)) if q != i]); a = rng.uniform(0, 2 * np.pi)
                c = xy[j] + (CUBE + rng.uniform(0.0003, 0.001)) * np.array([math.cos(a), math.sin(a)])
            else:
                c = rng.uniform(-TRAY + 0.006, TRAY - 0.006, 2)
            y = rng.uniform(0, math.pi / 2)
            if in_tray(c[None], np.array([y]))[0] and pair_ok(c[None], np.array([y]), xy, yaw, skip=i):
                xy[i], yaw[i] = c, y; break
    return pack(state, xy, yaw)


def components(xy, gap=0.0015):
    n = len(xy); lab = -np.ones(n, int); c = 0
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    for s in range(n):
        if lab[s] >= 0: continue
        st = [s]; lab[s] = c
        while st:
            u = st.pop()
            for v in np.nonzero((d[u] < CUBE + gap) & (lab < 0))[0]: lab[v] = c; st.append(v)
        c += 1
    return lab


def op_cluster_shift(state, rng, tries=300, max_move=0.04):
    xy, yaw = split(state); lab = components(xy)
    if lab.max() == 0: return op_rigid(state, rng)           # single blob: shift the whole thing
    comp = rng.choice(lab.max() + 1); m = lab == comp; ctr = xy[m].mean(0)
    for _ in range(tries):
        th = rng.uniform(-math.pi, math.pi) * rng.integers(0, 2); sh = rng.normal(0, max_move / 2, 2)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        xy2 = xy.copy(); yaw2 = yaw.copy(); xy2[m] = (xy[m] - ctr) @ R.T + ctr + sh; yaw2[m] = yaw[m] + th
        if legal(xy2, yaw2): return pack(state, xy2, yaw2)
    return None


def op_rigid(state, rng, tries=100):
    xy, yaw = split(state); ctr = xy.mean(0)
    for _ in range(tries):
        th = rng.uniform(-math.pi, math.pi)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        xy2 = (xy - ctr) @ R.T; yaw2 = yaw + th
        ext = HALF * (np.abs(np.cos(yaw2)) + np.abs(np.sin(yaw2)))
        lo = -(TRAY - TRAY_MARGIN) + (-(xy2 - ext[:, None])).max(0) * 0 - (xy2 - ext[:, None]).min(0) * 0   # placeholder, computed below
        lo = (-(TRAY - TRAY_MARGIN) - (xy2 - ext[:, None]).min(0)); hi = ((TRAY - TRAY_MARGIN) - (xy2 + ext[:, None]).max(0))
        if (hi < lo).any(): continue
        sh = rng.uniform(lo, hi); xy3 = xy2 + sh
        if legal(xy3, yaw2): return pack(state, xy3, yaw2)
    return None


OPS = {"jitter_small": lambda s, r: op_jitter(s, r, 0.002, 0.3), "jitter_large": lambda s, r: op_jitter(s, r, 0.006, 0.8),
       "relocate3": lambda s, r: op_relocate(s, r, 3), "relocate6": lambda s, r: op_relocate(s, r, 6),
       "cluster_shift": op_cluster_shift, "rigid": op_rigid}


def perturb_state(state, rng, ops=None):
    """state (n,7) float array -> (new_state or None, recipe). One random op, or the named sequence."""
    names = ops or [rng.choice(list(OPS))]
    out = state
    for nm in names:
        out = OPS[nm](out, rng)
        if out is None: return None, names
    xy, yaw = split(out)
    return (out if legal(xy, yaw) else None), names


# ------------------------------------------------------------------ sim re-settle
def build_sim(n_envs=32):
    sys.path.insert(0, ".")
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    physics = dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config("simple_mpc/config/config_oracle.yaml"), physics)
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = n_envs
    env = GenesisOracleEnv(cfg, n_envs=n_envs); apply_physics(env, physics); sim = env._sim
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    import torch; torch.set_default_device("cpu")
    return env, sim


def settle_in_sim(sim, states):
    """states (E,n,7) numpy/tensor (E == sim._n_envs) -> settled (E,n,7) tensor + per-env report dict list."""
    import torch, genesis as gs
    st = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    sim.set_particle_state(st[:, :, :3].to(gs.device).contiguous(), st[:, :, 3:7].to(gs.device).contiguous())
    sim.update_material_state()
    out = sim._particle_state[:, :, :7].detach().float().cpu().clone()
    rep = []
    for e in range(len(out)):
        xy, yaw = out[e, :, :2].numpy().astype(float), yaw_of(out[e, :, 3:7].numpy().astype(float))
        rep.append(dict(layer0=bool((np.abs(out[e, :, 2].numpy() - Z0) < 0.5 * CUBE).all()), in_tray=bool(in_tray(xy, yaw, 0.0).all()),
                        overlap=bool(any_overlap_matrix(xy, yaw, CUBE, tol=LEGAL_TOL).any()),
                        drift_mm_max=float(np.linalg.norm(xy - st[e, :, :2].numpy(), axis=1).max() * 1000),
                        drift_mm_mean=float(np.linalg.norm(xy - st[e, :, :2].numpy(), axis=1).mean() * 1000)))
    return out, rep


# ------------------------------------------------------------------ source states
def source_states(kind="scatter", split_dir="train_v2_clean", n=512, seed=0):
    """Unique-ish DS-0015 chain states (every chain step) of the requested start kind; returns (n,20,7) numpy."""
    import glob, torch
    rng = np.random.default_rng(seed); pool = []
    for f in sorted(glob.glob(f"Genesis/data/narrow_l20_n20/{split_dir}/_*_data.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        sk = np.array(d["start_kind"]); ok = (sk == kind) if kind != "any" else np.ones(len(sk), bool)
        ok &= d["valid"].bool().numpy() if "valid" in d else True
        pool.append(d["states"][torch.from_numpy(ok)].float().numpy())
    pool = np.concatenate(pool); return pool[rng.choice(len(pool), min(n, len(pool)), replace=False)]


def stats(S):
    """Distribution descriptors of a set of states (N,n,7): nn distance, contacts, centroid spread, pile bbox, wall gap."""
    nn, con, wall, bb = [], [], [], []
    for s in S:
        xy = s[:, :2].astype(float); d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy))
        nn.append(d.min(1).mean() * 1000); con.append(((d < CUBE + 0.001).sum() / 2)); wall.append((TRAY - np.abs(xy).max(1)).min() * 1000)
        bb.append(np.prod(xy.max(0) - xy.min(0)) * 1e4)
    f = lambda v: [round(float(np.mean(v)), 2), round(float(np.std(v)), 2)]
    return dict(mean_nn_dist_mm=f(nn), contacts=f(con), min_wall_gap_mm=f(wall), bbox_area_cm2=f(bb))


def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--sanity", action="store_true"); ap.add_argument("--sim", action="store_true")
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--kind", default="scatter"); ap.add_argument("--out", default="experiments/EXP-0073-active-data-mining/runs/active_pilot/")
    a = ap.parse_args(); rng = np.random.default_rng(0)
    src = source_states(a.kind, n=a.n); print("source", src.shape, flush=True)
    res = {"source_legal_frac": float(np.mean([legal(*split(s)) for s in src]))}
    per = {}; kept = {}
    for nm in list(OPS):
        outs = [perturb_state(s, rng, [nm])[0] for s in src]; ok = [o is not None for o in outs]
        good = [o for o in outs if o is not None]
        moved = [np.linalg.norm(o[:, :2] - s[:, :2], axis=1) * 1000 for o, s in zip(outs, src) if o is not None]
        per[nm] = dict(success=float(np.mean(ok)), illegal_after=float(np.mean([not legal(*split(g)) for g in good])) if good else None,
                       mean_move_mm=float(np.mean([m.mean() for m in moved])) if moved else None, max_move_mm=float(np.mean([m.max() for m in moved])) if moved else None,
                       frac_cubes_moved_gt2mm=float(np.mean([(m > 2).mean() for m in moved])) if moved else None)
        kept[nm] = np.stack(good) if good else None
        print(nm, per[nm], flush=True)
    res["ops"] = per; res["stats_source"] = stats(src)
    res["stats_perturbed_all_ops"] = stats(np.concatenate([v for v in kept.values() if v is not None]))
    for nm, v in kept.items():
        if v is not None: res[f"stats_{nm}"] = stats(v)
    if a.sim:
        env, sim = build_sim(32); E = 32; reps = {}; settled = {}
        for nm, v in kept.items():
            if v is None or len(v) < E: continue
            t0 = time.time(); out, rep = settle_in_sim(sim, v[:E]); settled[nm] = (v[:E], out.numpy())
            reps[nm] = dict(layer0=float(np.mean([r["layer0"] for r in rep])), in_tray=float(np.mean([r["in_tray"] for r in rep])),
                            overlap=float(np.mean([r["overlap"] for r in rep])), drift_mm_max_mean=float(np.mean([r["drift_mm_max"] for r in rep])),
                            drift_mm_mean=float(np.mean([r["drift_mm_mean"] for r in rep])), seconds=round(time.time() - t0, 1)); print("settle", nm, reps[nm], flush=True)
        res["settle"] = reps
        # baseline: re-settling UNPERTURBED source states (drift floor)
        out, rep = settle_in_sim(sim, src[:E]); res["settle_unperturbed_source"] = dict(drift_mm_mean=float(np.mean([r["drift_mm_mean"] for r in rep])), drift_mm_max_mean=float(np.mean([r["drift_mm_max"] for r in rep])), layer0=float(np.mean([r["layer0"] for r in rep])))
        np.savez_compressed(a.out + f"perturbed_{a.kind}_pilot.npz", **{f"{nm}_pre": v[0] for nm, v in settled.items()}, **{f"{nm}_post": v[1] for nm, v in settled.items()}, src=src[:E])
        env.destroy()
    json.dump(res, open("experiments/EXP-0073-active-data-mining/results/active_perturb_sanity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
