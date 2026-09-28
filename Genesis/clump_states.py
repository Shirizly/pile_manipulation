"""Genesis/clump_states.py -- single-layer CLUMP start states for n-cube scenes.

Moved from experiments/EXP-0049-sampler-study/code/sampler_study.py (where it was built
and validated: 0/32 proposals rejected) so collection drivers can reuse it.
`make_clump_states(sim, n, seed)` proposes 1-3 tight lattice patches per env (pitch = cube
width + 0.5 mm), writes them with set_particle_state, settles once, and keeps only
states whose every cube stays in layer 0 and inside the tray.
`clump_starts(sim, n, rng)` is the chain_collection.py --clump-fn adapter.
"""
from __future__ import annotations
import math
import numpy as np
import torch

CUBE = 0.005
Z0 = 0.012471          # settled layer-0 cube-centre height (DS-0006 states)


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


def clump_starts(sim, n: int, rng) -> torch.Tensor:
    """--clump-fn adapter for Genesis/chain_collection.py: n settled clump states."""
    states, _ = make_clump_states(sim, n, seed=int(rng.integers(0, 2**31 - 1)))
    return states
