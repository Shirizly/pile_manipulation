"""EXP-0021 arm `descriptor-readout` -- the descriptor arm with a TRAINED value
readout instead of the degenerate point-mass field sample.

Plugin for `build_cache_9cell.py --plugin`. It does NOT replace the existing
`descriptor` arm: both rows are produced, and their difference is the readout's
contribution.

Why a different basis from the `descriptor` arm
-----------------------------------------------
`ValueReadout` compares a STATE embedding against a GOAL embedding, so the two
must live in the same frame. MODEL-0002's 94-dim D-all-local basis is
PUSH-FRAME and a world-frame goal cannot be expressed in it (REGISTER C-014:
no cross-frame reframing primitive exists for `moments2`/`band_mass`/`dft` --
only COM has one, which is exactly why the point-mass readout existed). This
arm therefore runs on the 87-dim WORLD-FRAME basis
`dmdc_baseline.occupancy_descriptors(occ, n_fourier=8)`, in which the predicted
descriptor feeds the readout directly and no reframing is needed.

Dynamics: `experiments/temp/exp0021-desc-readout/desc87_world_switched.pt`,
6 equal-width push-length bins over [0, 0.080] m
(`Baselines/LinearForesight/model.py::bin_index`), fit on the shared split
`overnight_randlen_train` (192 files / 98,304 rows), held out on
`overnight_randlen_test`. The ACTION-AUGMENTED operator is used:
`phi1 ~ [A_b B_b] [phi0 ; u]`. A state-only operator `phi1 ~ A_b phi0` is
USELESS as a ranker in the world frame -- `phi0` is identical for every
candidate of a slate, so it can emit at most 6 distinct scores per slate. The
push frame hid this because the frame is itself a function of the action.

Goal embedding: the cell's mask -> `mask_to_configuration` (n_objects=20,
regenerated with post-28271c09 code) -> rasterised -> the same 87-dim basis.
This is the goal representation the readout was trained on. `N_GOAL_DRAWS`
independent legal configurations are drawn and their predicted values averaged,
so the score does not hinge on one placement sample.

SENSE: the readout predicts the value function's own value, so the arm returns
`cell.sign * (h(phi1_hat, g) - h(phi0, g))` -- sign-corrected to the cache's
COST convention (lower is better), exactly like every other arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
TEMP = REPO / "experiments/temp/exp0021-desc-readout"
for p in (str(REPO), str(REPO / "experiments/EXP-0017-value-readout-instrument/code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dmdc_baseline import occupancy_descriptors                      # noqa: E402
from transforms.functional import particles_to_occupancy             # noqa: E402
from Baselines.LinearForesight.model import bin_index                # noqa: E402
from Baselines.common.goal_configs import mask_to_configuration      # noqa: E402
from value_readout import ValueReadout                               # noqa: E402

CKPT = TEMP / "desc87_world_switched.pt"
N_FOURIER = 8
N_GOAL_DRAWS = 3
N_OBJECTS = 20
# which persisted readout to use; overridable with ARM_READOUT_{CAPACITY,MODE}
CAPACITY = "mlp"
MODE = "diff"


def _readout_path(capacity, mode, value_fn):
    return TEMP / f"readout__{capacity}__{mode}__{value_fn}.joblib"


def _action_features(p_start, p_stop, angle):
    d = p_stop - p_start
    L = d.norm(dim=-1, keepdim=True)
    return torch.cat([p_start, p_stop, d, torch.cos(angle)[:, None],
                      torch.sin(angle)[:, None], L, torch.ones_like(L)], dim=1)


def make_arm(capacity=CAPACITY, mode=MODE, device="cpu"):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ops = [o.float() for o in ck["ops_action_aug"]]
    bin_edges = torch.as_tensor(ck["bin_edges"]).float()
    cfg = ck["config"]
    bounds, grid, radius = cfg["bounds"], cfg["grid"], cfg["footprint_radius"]
    readouts, goal_emb = {}, {}

    def phis(ctx):
        """(phi0, phi1_hat) for the whole pool, memoised on ctx."""
        key = "desc87_readout"
        if key not in ctx.cache:
            occ0 = ctx.occ0.detach().to("cpu").float()
            phi0 = occupancy_descriptors(occ0, n_fourier=N_FOURIER)
            u = _action_features(ctx.p_start.float(), ctx.p_stop.float(),
                                 ctx.angles.float())
            x = torch.cat([phi0, u], dim=1)
            b = bin_index(ctx.length_m.float(), bin_edges)
            phi1 = torch.empty_like(phi0)
            for k in range(len(ops)):
                m = b == k
                if m.any():
                    phi1[m] = (ops[k] @ x[m].T).T
            ctx.cache[key] = (phi0.numpy().astype(np.float32),
                              phi1.numpy().astype(np.float32))
        return ctx.cache[key]

    def goal_phi(cell):
        if cell.goal not in goal_emb:
            mask = cell.mask.detach().to("cpu").numpy().astype(bool)
            rows = []
            for s in range(N_GOAL_DRAWS):
                res = mask_to_configuration(mask, n_objects=N_OBJECTS,
                                            bounds=bounds, seed=1000 + s)
                poses = res.poses if hasattr(res, "poses") else res.config
                xyz = torch.as_tensor(np.asarray(poses)[:, :3], dtype=torch.float32)[None]
                occ = particles_to_occupancy(xyz, bounds, (grid, grid),
                                             footprint_radius=radius)
                rows.append(occupancy_descriptors(occ, n_fourier=N_FOURIER)[0])
            goal_emb[cell.goal] = torch.stack(rows).numpy().astype(np.float32)
        return goal_emb[cell.goal]

    def score_fn(ctx, cell):
        if cell.value_fn not in readouts:
            p = _readout_path(capacity, mode, cell.value_fn)
            if not p.exists():
                raise FileNotFoundError(
                    f"no persisted readout for {cell.value_fn} at {p} -- fit it with "
                    f"experiments/temp/exp0021-desc-readout/stage3_fit_readout.py --persist")
            readouts[cell.value_fn] = ValueReadout.load(p)
        ro = readouts[cell.value_fn]
        phi0, phi1 = phis(ctx)
        g = goal_phi(cell)                       # (N_GOAL_DRAWS, 87)
        v1 = ro.predict(phi1, g).mean(axis=1)    # (n,)
        v0 = ro.predict(phi0, g).mean(axis=1)
        return cell.sign * torch.from_numpy((v1 - v0).astype(np.float32))

    return score_fn


def register(builder, device="cpu", capacity=CAPACITY, mode=MODE):
    builder.register_score_arm("descriptor-readout",
                               make_arm(capacity=capacity, mode=mode, device=device))
    print(f"[plugin] registered `descriptor-readout` "
          f"(87-dim world-frame, action-augmented switched op, "
          f"{capacity}/{mode} ValueReadout)", flush=True)
