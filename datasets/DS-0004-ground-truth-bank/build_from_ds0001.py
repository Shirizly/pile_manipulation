"""Seed DS-0004 (the ground-truth bank) with DS-0001's 20 x 1000 simulated pushes.

sim_path "binned_collection": these outcomes came from the corpus collector,
NOT from GenesisOracleEnv.rollout_candidates, and the bank will never serve
them to a rollout_candidates lookup (EXP-0023: r = 0.959 between the two paths).

Actions are stored as [sx, sy, ex, ey] (metres, = `binned_pool_cache.py`'s
`actions` and EXP-0023's pool rows); the corpus's explicit yaw is kept per row
alongside in `extras.pt` rather than in the key.

Known-number check: dv = lyapunov(corner) recomputed from the STORED final
states must reproduce experiments/temp/binned-pools/dv_cache_corner.pt's
dv_true to float tolerance.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/probes"))
from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.gt_bank import GroundTruthBank, state_key
from utils import git_provenance

CORPUS = REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm/step0.pt"
CACHE = REPO / "experiments/temp/binned-pools/dv_cache_corner.pt"
ROOT = REPO / "datasets/DS-0004-ground-truth-bank/data"
FP = {"corpus": "DS-0001", "path": "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm",
      "collector": "Genesis/binned_slate_collection.py", "n_particles": 20}


def main():
    from binned_pool_cache import occ_of, GRID
    d = torch.load(CORPUS, map_location="cpu", weights_only=False)
    bank = GroundTruthBank(ROOT)
    si = d["slate_idx"].long()
    acts = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
    keys = {}
    n_added = 0
    for s in si.unique().tolist():
        r = (si == s).nonzero(as_tuple=True)[0]
        st = d["states"][r[0]].float()
        assert (d["states"][r].float() - st).abs().max() == 0, "slate rows must share one start state"
        n_added += bank.add(st, acts[r], d["states_"][r].float(), "binned_collection", FP, "DS-0001")
        keys[int(s)] = state_key(st)

    # known-number check against the dv cache
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dw = lyapunov_weights((GRID, GRID), "corner", dev)
    worst = 0.0
    for s, k in keys.items():
        r = (c["ep"] == s).nonzero(as_tuple=True)[0]
        rec = bank.record("binned_collection", k)
        hit, fin = bank.lookup(rec["state"], c["actions"][r], "binned_collection")
        assert hit.all()
        v0 = lyapunov(occ_of(rec["state"][None]), dw).cpu()
        dv = lyapunov(occ_of(fin), dw).cpu() - v0
        worst = max(worst, float((dv - c["dv"]["corner"]["dv_true"][r]).abs().max()))
    print(f"added {n_added} rows over {len(keys)} states; max |dv - cached dv_true| = {worst:.2e}")
    assert worst < 1e-5
    torch.save({"slate_to_state_key": keys, "angles": d["angles"], "slate_idx": si,
                "provenance": git_provenance()}, ROOT / "binned_collection" / "extras.pt")


if __name__ == "__main__":
    main()
