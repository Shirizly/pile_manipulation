"""Shared helpers for the EXP-0072 ceiling study (perturbation, rasters per model frame, data loading)."""
import glob, sys
import numpy as np, torch
sys.path.insert(0, ".")
D = "Genesis/data/narrow_l20_n20/"
SIZES = [[0.005] * 3] * 20
OUT = "experiments/EXP-0072-zoom-window-nfd/results/ceiling/"
YAW_DEG_PER_MM = 2.0     # yaw jitter std (deg) = 2 deg x xy std in mm (0.5 mm -> 1 deg, the EXP-0059 chaos_floor yaw)


def load_chains():
    """DS-0016 test_chains_v2_clean, flattened, rows sorted by (chain id, step). Returns dict of tensors + chain table."""
    fs = sorted(glob.glob(D + "test_chains_v2_clean/_*_data.pt")); rows = []
    for fi, f in enumerate(fs):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for i in range(len(d["states"])):
            rows.append((fi, int(d["chain_env"][i]), int(d["chain_step"][i]), d, i))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    g = lambda k: torch.stack([r[3][k][r[4]] for r in rows])
    out = {k: g(k) for k in ["states", "states_", "p_starts", "p_stops", "angles"]}
    out["chain"] = torch.tensor([r[0] * 1000 + r[1] for r in rows]); out["step"] = torch.tensor([r[2] for r in rows])
    out["kind"] = np.array([r[3]["start_kind"][r[4]] for r in rows])
    return out


def perturb(states, xy_mm, rng):
    from model.retrieval.frame import yaw_from_quat, yaw_to_quat
    out = states.clone()
    if xy_mm > 0:
        out[..., 0:2] += torch.from_numpy(rng.normal(0, xy_mm / 1000., size=out[..., 0:2].shape)).float()
        yaw = yaw_from_quat(out[..., 3:7]) + torch.from_numpy(rng.normal(0, np.deg2rad(YAW_DEG_PER_MM * xy_mm), size=out.shape[:2])).float()
        out[..., 3:7] = yaw_to_quat(yaw)
    return out
