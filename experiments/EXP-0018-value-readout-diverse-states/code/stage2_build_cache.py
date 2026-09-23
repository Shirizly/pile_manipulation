"""Stage 2 (EXP-0018) -- build the state/goal feature + value cache.

States: AFTER-SWEEP states (`states_`) sampled across FIVE source corpora, so the
training set spans spawn modes, particle counts and push lengths instead of the
near-duplicate post-push states of one slate pool (the EXP-0017 failure mode):

  randlen            Genesis/data/overnight_randlen_train/{mixed,piled,scattered}_n{20,50}
  sean_inbetween     Genesis/data/Sean/inbetween-*/inbetween/cube/...
  sean_piled         Genesis/data/Sean/piled-*/piled/cube/...
  sean_scattered     Genesis/data/Sean/scattered-*/scattered/cube/...
  slates_multistep   Genesis/data/slates_multistep/{n20_L20mm,n20_L40mm}

Rasterisation is ONE function for every corpus and for the goal configurations:
`transforms.functional.particles_to_occupancy` with footprint_radius = 0.5*CUBE_SIZE/pitch.
Embedding: `dmdc_baseline.occupancy_descriptors(n_fourier=8)` -> 87 dims.

Groups (near-duplicate protection): `group_id` is the source file for randlen/Sean
(independent transitions inside a file, but files share a spawn state library) and
`ms:<corpus>:s<slate_idx>` for slates_multistep, whose 3 steps are ONE trajectory
and therefore near-duplicates. Every split and every inner CV fold is built over
group_id, never over rows.

Output: experiments/temp/exp0018-value-readout/cache.pt
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from dmdc_baseline import occupancy_descriptors                       # noqa: E402
from transforms.functional import particles_to_occupancy              # noqa: E402
from Baselines.common.goals import dist_field_from_mask               # noqa: E402
from Baselines.common.goal_configs import DEFAULT_BOUNDS, CUBE_SIZE   # noqa: E402
from utils import git_provenance                                      # noqa: E402

OUT = REPO / "experiments/temp/exp0018-value-readout"
OUT.mkdir(parents=True, exist_ok=True)
GOALS = REPO / "experiments/temp/goal-states/dataset_v2.pt"

BOUNDS = DEFAULT_BOUNDS
GRID = 64
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
N_FOURIER = 8
SEED = 0

PER_FILE = {"randlen": 40, "sean": 8, "ms": 20}


def occ_of(poses):
    return particles_to_occupancy(poses[..., :3].float(), BOUNDS, (GRID, GRID),
                                  footprint_radius=RADIUS)


def collect():
    rng = np.random.default_rng(SEED)
    recs = []   # (corpus, group_id, file, row_idx)
    occs = []
    push_len = []

    def take(path, corpus, group_id, per_file):
        d = torch.load(path, weights_only=False)
        s_ = d["states_"]
        n = s_.shape[0]
        k = min(per_file, n)
        idx = rng.choice(n, k, replace=False)
        idx.sort()
        pl = (d["p_stops"] - d["p_starts"])[idx][:, :2].norm(dim=-1).numpy()
        occs.append(occ_of(s_[idx]))
        for j, r in enumerate(idx):
            recs.append((corpus, group_id, str(path), int(r)))
        push_len.extend(pl.tolist())

    t0 = time.time()
    # --- randlen -----------------------------------------------------------
    for f in sorted(glob.glob(str(REPO / "Genesis/data/overnight_randlen_train/*/*_data.pt"))):
        if f.endswith("_failed.pt"):
            continue
        take(f, "randlen", f, PER_FILE["randlen"])
    print(f"randlen: {len(recs)} states ({time.time()-t0:.0f}s)", flush=True)

    # --- Sean --------------------------------------------------------------
    for mode in ("inbetween", "piled", "scattered"):
        n0 = len(recs)
        files = sorted(glob.glob(str(REPO / f"Genesis/data/Sean/{mode}-*/{mode}/cube/**/*_data.pt"),
                                 recursive=True))
        files = [f for f in files if not f.endswith("_failed.pt")]
        for f in files:
            take(f, f"sean_{mode}", f, PER_FILE["sean"])
        print(f"sean_{mode}: {len(files)} files -> {len(recs)-n0} states "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- slates_multistep --------------------------------------------------
    for corpus in ("n20_L20mm", "n20_L40mm"):
        n0 = len(recs)
        root = REPO / f"Genesis/data/slates_multistep/{corpus}"
        man = json.load(open(root / "manifest.json"))
        for b in man["batches"]:
            f = root / f"_{b['batch_idx']}_data.pt"
            if not f.exists():
                continue
            take(f, "slates_multistep", f"ms:{corpus}:s{b['slate_idx']}", PER_FILE["ms"])
        print(f"slates_multistep/{corpus}: {len(recs)-n0} states ({time.time()-t0:.0f}s)", flush=True)

    occ = torch.cat(occs)
    return occ, recs, np.asarray(push_len, dtype=np.float32)


def main():
    occ, recs, push_len = collect()
    S = occ.shape[0]
    corpus = np.array([r[0] for r in recs])
    group = np.array([r[1] for r in recs])
    print(f"S={S} states; corpora={dict(zip(*np.unique(corpus, return_counts=True)))}", flush=True)
    print(f"push length mm: min={push_len.min()*1e3:.1f} max={push_len.max()*1e3:.1f} "
          f"mean={push_len.mean()*1e3:.1f}", flush=True)
    print(f"n unique groups = {len(np.unique(group))}", flush=True)

    t0 = time.time()
    phi_state = torch.cat([occupancy_descriptors(occ[i:i + 2000], n_fourier=N_FOURIER)
                           for i in range(0, S, 2000)]).numpy().astype(np.float32)
    print(f"phi_state {phi_state.shape} in {time.time()-t0:.0f}s", flush=True)

    gd = torch.load(GOALS, weights_only=False)
    masks = gd["masks"].numpy()
    configs = gd["configs"]
    G, K = configs.shape[0], configs.shape[1]
    t0 = time.time()
    phi_goal = np.zeros((G, phi_state.shape[1]), dtype=np.float32)
    for lo in range(0, G, 100):
        hi = min(lo + 100, G)
        occ_k = occ_of(configs[lo:hi].reshape(-1, 20, 7)).reshape(hi - lo, K, GRID, GRID).mean(1)
        phi_goal[lo:hi] = occupancy_descriptors(occ_k, n_fourier=N_FOURIER).numpy()
    print(f"phi_goal {phi_goal.shape} in {time.time()-t0:.0f}s (goal = mean occupancy over "
          f"{K} LEGAL 20-cube configurations, not a solid mask)", flush=True)

    # --- value targets, computed from the goal MASKS -----------------------
    # All three are expressible from occ @ [d ; mask]:
    #   lyapunov            = (occ@d)/mass
    #   mass_in_region      = occ@mask
    #   signed_mass_in_reg. = 2*(occ@mask) - mass
    t0 = time.time()
    flat = occ.reshape(S, -1).numpy().astype(np.float32)
    mass = flat.sum(1)
    D = np.stack([dist_field_from_mask(masks[g]).reshape(-1) for g in range(G)], 1).astype(np.float32)
    M = masks.reshape(G, -1).T.astype(np.float32)
    A = flat @ D
    B = flat @ M
    values = {
        "lyapunov": A / np.maximum(mass, 1e-6)[:, None],
        "mass_in_region": B,
        "signed_mass_in_region": 2 * B - mass[:, None],
    }
    print(f"values {S}x{G} x3 in {time.time()-t0:.0f}s", flush=True)
    for k, v in values.items():
        print(f"  {k:22s} mean={v.mean():+.4f} std={v.std():.4f}", flush=True)

    torch.save({
        "phi_state": phi_state, "phi_goal": phi_goal, "values": values,
        "corpus": corpus, "group": group, "push_len_m": push_len,
        "state_index": recs,          # (corpus, group, file, row) per state -- reproducible
        "n_fourier": N_FOURIER, "grid": GRID, "bounds": BOUNDS,
        "footprint_radius_voxels": RADIUS,
        "rasteriser": "transforms.functional.particles_to_occupancy",
        "embedding": "dmdc_baseline.occupancy_descriptors(n_fourier=8)",
        "goals_dataset": str(GOALS), "seed": SEED, "per_file": PER_FILE,
        "provenance": git_provenance(),
    }, OUT / "cache.pt")
    print(f"saved {OUT/'cache.pt'}", flush=True)


if __name__ == "__main__":
    main()
