"""Particle-in / particle-out (and, optionally, occupancy-out) retrieval
predictors (EXP-0059).

Core contract every predictor here follows -- deliberately NOT
`simple_mpc.adapters.OCC_ADAPTERS`'s occupancy-in/occupancy-out contract:

    predict_particles(states0 (B, n, 7), p_start (B, >=2), p_stop (B, >=2))
        -> states1_pred (B, n, 7)

A retrieval model needs the query's raw cube positions to retrieve against;
an occupancy grid has already thrown that away. Keeping the output in
particle space too means a multi-step rollout can compose predictions in the
space it retrieves in.

`RetrievalPredictor` additionally exposes an OPTIONAL

    predict_occ(states0, p_start, p_stop) -> occ (B, 64, 64)

for the "hedged" k-NN occupancy-average aggregations (`occ_mean`,
`occ_weighted_mean`), which have no single well-defined PARTICLE state to
return (that is the whole point of hedging across k disagreeing
hypotheses).

**Hedged occupancy is scoring-only, never a state the model commits to**
(2026-09-28 user guidance): `predict_particles` ALWAYS returns a single
CLEAN, feasible-as-possible per-cube state (the top-1 neighbour's transfer,
or a per-cube median/weighted-mean across k for the `cube_*` aggregations)
-- this is what rollout composition and `slateN` use, never the hedged
occupancy blend. `eval_retrieval.py` prefers `predict_occ` via `hasattr`
ONLY when scoring single-step `accuracy_1` (exactly the optional-method
pattern `WarpedNFDPredictor.predict_occ_canonical` already uses in this
codebase, see docs/CODEMAP.md "Canonical-frame scoring"); its rollout and
slateN code paths call `predict_particles` unconditionally. Per-cube overlap
resolution (making the clean state not just feasible-ish but strictly
non-penetrating) is NOT implemented -- a known, documented limitation, not
an oversight (see the predictor class docstring below).

Every predictor also optionally exposes

    predict_particles_with_confidence(states0, p_start, p_stop)
        -> (states1_pred, {"top1_dist": (B,), "knn_disagreement": (B,)})

a per-query CONFIDENCE signal computed FOR FREE alongside the clean
prediction (no second bank search) -- for finding actions the bank predicts
least well, not for scoring.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from scipy.optimize import linear_sum_assignment

from .bank import TransitionBank
from .frame import (world_to_push_frame, push_frame_to_world,
                    yaw_from_quat, yaw_to_quat, wrap_angle)
from .distance import (DistanceConfig, query_points_and_weights,
                       bank_points_and_weights, topk_search)
from .interaction import interaction_set

DEFAULT_DISTANCE_GATE = 0.006  # metres; see RetrievalPredictor's docstring, ISS-010/EXP-0059 fix

AGGREGATIONS = ("nn1", "occ_mean", "occ_weighted_mean", "cube_median", "cube_weighted_mean")


class PersistencePredictor:
    """states1_pred = states0. The reference floor: `eval_narrow.py`'s own
    `acc()` makes this accuracy=0 EXACTLY, by construction (pred == prev, so
    numerator == denominator on the same swept region, for ANY region) --
    this is a plumbing check, not a claim this predictor is good."""
    name = "persistence"

    def predict_particles(self, states0: torch.Tensor, p_start: torch.Tensor,
                          p_stop: torch.Tensor) -> torch.Tensor:
        return states0.clone()

    def predict_particles_with_confidence(self, states0: torch.Tensor, p_start: torch.Tensor,
                                          p_stop: torch.Tensor):
        """No retrieval happens, so both confidence signals are degenerate
        (0.0): kept for interface symmetry with `RetrievalPredictor`, not
        because persistence has anything informative to say about
        confidence."""
        B = states0.shape[0]
        z = torch.zeros(B)
        return states0.clone(), {"top1_dist": z, "knn_disagreement": z.clone()}


class RetrievalPredictor:
    """Whole-bank exhaustive k-NN retrieval with a configurable push-frame
    distance (`distance.DistanceConfig`) and a choice of aggregation over
    the k neighbours.

    Aggregations
    ------------
    nn1                  k should be 1; direct transfer of the single
                         nearest neighbour's per-cube displacement/yaw
                         (Hungarian-matched, gated by the donor cube's own
                         `moved` flag). The "clean" baseline.
    cube_median          k >= 1; per query cube, the MEDIAN (over the k
                         Hungarian-matched neighbours) of the transferred
                         push-frame displacement/yaw-delta. Still a single,
                         well-defined state ("clean-ish": robust to one bad
                         neighbour without blurring the prediction the way
                         occupancy-averaging does).
    cube_weighted_mean   k >= 1; like `cube_median` but a distance-weighted
                         MEAN (softmax(-dist / `weight_temperature`)).
    occ_mean             k >= 1; rasterises each of the k neighbour-
                         transferred candidate states and averages the
                         occupancies (the "hedged" variant design doc
                         Section 1 / the designer's note #2 describe --
                         rewards spreading probability mass instead of
                         committing to one hypothesis). Only meaningful
                         through `predict_occ`; `predict_particles` falls
                         back to the top-1 neighbour for this config so
                         rollout composition stays well-defined.
    occ_weighted_mean    like `occ_mean` but distance-weighted.

    Known limitation (documented, not fixed here): the clean per-cube
    transfer does not check for cube-cube overlap in the predicted state --
    "clean" here means "a single well-defined state", not "verified
    non-penetrating". A post-hoc legality repair (e.g. `Baselines.common.
    cube_overlap`) would be the natural next step if overlap turns out to
    matter for a downstream consumer (e.g. an MPC rollout).

    **Curated matching + distance gate** (2026-09-28, coordinator follow-up
    B/C, fixing a real bug: `_transfer_one` used to Hungarian-match ALL n
    query cubes against ALL n donor cubes with no distance limit, so a query
    cube far outside the corridor could be paired with a moving donor cube
    and inherit a large, physically-nonsensical displacement -- visible in
    `retrieval_debug.py`'s `q0908` figure, a cube 20+ mm outside the swept
    corridor moving 23mm). Now:
      (1) both sides of the match are restricted to their own geometric
          interaction set (`interaction.interaction_set`, truth-free --
          computed on the query fresh every call; read from `bank.in_set`
          when the bank was built by `TransitionBank.load_curated` /
          DS-0014, else every donor cube is eligible, matching old
          behaviour for an uncurated bank);
      (2) every matched pair is additionally gated by `distance_gate`
          (default 6mm): a match farther apart than that transfers NOTHING
          (the query cube stays put), rather than the donor's raw
          displacement. Cubes outside the query's own interaction set are
          never matched at all and are always carried through unchanged.
    This changes `RetrievalPredictor`'s numeric output vs the pre-2026-09-28
    behaviour even against the OLD (uncurated) `artifacts/bank.pt` -- see
    `experiments/EXP-0059-*/LOG.md`'s post-fix rerun for exactly which
    recorded numbers moved and by how much.
    """
    def __init__(self, bank: TransitionBank, cfg: Optional[DistanceConfig] = None,
                k: int = 1, aggregation: str = "nn1", transfer_yaw: bool = True,
                weight_temperature: float = 0.01, bank_search_chunk: int = 1500,
                device: Optional[str] = None, distance_gate: float = DEFAULT_DISTANCE_GATE):
        assert aggregation in AGGREGATIONS, f"unknown aggregation {aggregation!r}"
        self.bank = bank
        self.cfg = cfg or DistanceConfig()
        self.k = k
        self.aggregation = aggregation
        self.transfer_yaw = transfer_yaw
        self.weight_temperature = weight_temperature
        self.bank_search_chunk = bank_search_chunk
        self.distance_gate = distance_gate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._bank_pts, self._bank_w, self._bank_valid = bank_points_and_weights(bank, self.cfg)
        if getattr(bank, "in_set", None) is not None:
            # Curated bank: never retrieve against, or match onto, a donor's
            # own uninvolved cubes either -- zero their search weight too.
            self._bank_w = self._bank_w * bank.in_set.to(self._bank_w.dtype)
        self.name = f"retrieval_k{k}_{aggregation}"

    # -- search --------------------------------------------------------
    def _search(self, states0: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor):
        q_pts, q_w, q_uv, q_valid = query_points_and_weights(states0, p_start, p_stop, self.cfg)
        push_len = (p_stop[:, :2] - p_start[:, :2]).norm(dim=-1)
        q_in_set = interaction_set(q_uv, push_len)   # (B, n) bool, truth-free, every call
        q_w = q_w * q_in_set.to(q_w.dtype)
        idx, dist = topk_search(q_pts, q_w, self._bank_pts, self._bank_w, self.cfg,
                                k=self.k, chunk=self.bank_search_chunk, device=self.device,
                                bank_valid=self._bank_valid)
        return idx, dist, q_uv, q_valid, q_in_set

    def _transfer_one(self, uv0_b: torch.Tensor, donor_i: int, query_in_set: torch.Tensor = None):
        """Hungarian-match query cubes in `query_in_set` (default: all n, the
        pre-fix behaviour) to bank row `donor_i`'s cubes in its own
        `bank.in_set` (default: all n); matches farther apart than
        `self.distance_gate` transfer nothing. -> (duv (n,2), dyaw (n,)) in
        the QUERY's own push frame, zero outside the matched/gated set."""
        donor_uv0 = self.bank.uv0[donor_i]
        donor_duv = self.bank.duv[donor_i]
        donor_dyaw = self.bank.dyaw[donor_i]
        donor_moved = self.bank.moved[donor_i]
        n = uv0_b.shape[0]
        duv = torch.zeros(n, 2)
        dyaw = torch.zeros(n)

        q_mask = query_in_set if query_in_set is not None else torch.ones(n, dtype=torch.bool)
        d_mask = (self.bank.in_set[donor_i] if getattr(self.bank, "in_set", None) is not None
                 else torch.ones(n, dtype=torch.bool))
        q_idx = q_mask.nonzero(as_tuple=True)[0]
        d_idx = d_mask.nonzero(as_tuple=True)[0]
        if len(q_idx) == 0 or len(d_idx) == 0:
            return duv, dyaw   # nothing on one side to match against -> everyone stays

        cost = torch.cdist(uv0_b[q_idx], donor_uv0[d_idx]).numpy()
        row, col = linear_sum_assignment(cost)
        gated = cost[row, col] <= self.distance_gate   # unmatched-by-distance -> no transfer
        row, col = row[gated], col[gated]
        if len(row) == 0:
            return duv, dyaw
        row_q = q_idx[row]
        col_d = d_idx[col]
        moved_matched = donor_moved[col_d]
        duv[row_q] = torch.where(moved_matched.unsqueeze(-1), donor_duv[col_d], torch.zeros_like(donor_duv[col_d]))
        if self.transfer_yaw:
            dyaw[row_q] = torch.where(moved_matched, donor_dyaw[col_d], torch.zeros_like(donor_dyaw[col_d]))
        return duv, dyaw

    def _aggregate_cube(self, uv0_b: torch.Tensor, neighbours: Sequence[int],
                        dists_b: torch.Tensor, query_in_set: torch.Tensor = None):
        cand_duv, cand_dyaw = [], []
        for j in neighbours:
            duv, dyaw = self._transfer_one(uv0_b, int(j), query_in_set)
            cand_duv.append(duv); cand_dyaw.append(dyaw)
        cand_duv = torch.stack(cand_duv)     # (k, n, 2)
        cand_dyaw = torch.stack(cand_dyaw)   # (k, n)
        if self.aggregation == "cube_median":
            return cand_duv.median(dim=0).values, cand_dyaw.median(dim=0).values
        w = torch.softmax(-dists_b / max(self.weight_temperature, 1e-9), dim=0)  # (k,)
        duv = (cand_duv * w.view(-1, 1, 1)).sum(dim=0)
        dyaw = (cand_dyaw * w.view(-1, 1)).sum(dim=0)
        return duv, dyaw

    def _apply_delta(self, states0_b, uv0_b, p_start_b, p_stop_b, duv, dyaw, yaw0_w_b):
        uv1 = uv0_b + duv
        xy1 = push_frame_to_world(uv1.unsqueeze(0), p_start_b, p_stop_b)[0]
        out = states0_b.clone()
        out[:, 0:2] = xy1
        if self.transfer_yaw:
            yaw1 = wrap_angle(yaw0_w_b + dyaw)
            out[:, 3:7] = yaw_to_quat(yaw1)
        return out

    # -- public API ------------------------------------------------------
    def predict_particles(self, states0: torch.Tensor, p_start: torch.Tensor,
                          p_stop: torch.Tensor) -> torch.Tensor:
        B, n, _ = states0.shape
        states0c = states0.cpu()
        p_start_c, p_stop_c = p_start.cpu().float(), p_stop.cpu().float()
        idx, dist, uv0, q_valid, q_in_set = self._search(states0c, p_start_c, p_stop_c)
        yaw0_w = yaw_from_quat(states0c[..., 3:7])
        out = states0c.clone()
        for b in range(B):
            if not bool(q_valid[b]):
                continue  # no query cube near this push's swath: leave the row in place
            neighbours = idx[b]
            if self.aggregation in ("nn1", "occ_mean", "occ_weighted_mean"):
                duv, dyaw = self._transfer_one(uv0[b], int(neighbours[0]), q_in_set[b])
            else:
                duv, dyaw = self._aggregate_cube(uv0[b], neighbours, dist[b], q_in_set[b])
            out[b] = self._apply_delta(states0c[b], uv0[b], p_start_c[b:b + 1],
                                       p_stop_c[b:b + 1], duv, dyaw, yaw0_w[b])
        return out

    def predict_particles_with_confidence(self, states0: torch.Tensor, p_start: torch.Tensor,
                                          p_stop: torch.Tensor):
        """The CLEAN per-cube transfer (identical to `predict_particles`),
        plus a per-query confidence signal computed from the SAME retrieval
        search (no second bank search -- this is the "cheap" contract):

          * `top1_dist` (B,): the nearest neighbour's chamfer distance.
            Large -> nothing much like this query exists in the bank.
            `+inf` for a query with no cube in the window at all (the row
            that `predict_particles` leaves unchanged).
          * `knn_disagreement` (B,): only informative for `k > 1` (0.0 at
            k=1, nothing to disagree with): the mean, over query cubes,
            of the STD across the k neighbours' Hungarian-matched
            transferred displacement vectors. Large -> the k nearest
            transitions imply meaningfully different outcomes for this
            push (a geometric proxy for "the local dynamics here are not
            well pinned down by the bank", independent of and cheaper than
            actually simulating k candidates).

        Intended for finding actions the bank predicts least well (large
        `top1_dist` and/or `knn_disagreement`), not for scoring -- neither
        signal feeds into `accuracy_1`/`rollout`/`slateN`.
        """
        B, n, _ = states0.shape
        states0c = states0.cpu()
        p_start_c, p_stop_c = p_start.cpu().float(), p_stop.cpu().float()
        idx, dist, uv0, q_valid, q_in_set = self._search(states0c, p_start_c, p_stop_c)
        yaw0_w = yaw_from_quat(states0c[..., 3:7])
        out = states0c.clone()
        top1_dist = dist[:, 0].clone()
        disagreement = torch.zeros(B)
        for b in range(B):
            if not bool(q_valid[b]):
                top1_dist[b] = float("inf")
                continue
            neighbours = idx[b]
            if self.k > 1:
                cand_duv = torch.stack([self._transfer_one(uv0[b], int(j), q_in_set[b])[0]
                                       for j in neighbours])  # (k,n,2)
                disagreement[b] = float(cand_duv.std(dim=0, unbiased=False).norm(dim=-1).mean())
            if self.aggregation in ("nn1", "occ_mean", "occ_weighted_mean"):
                duv, dyaw = self._transfer_one(uv0[b], int(neighbours[0]), q_in_set[b])
            else:
                duv, dyaw = self._aggregate_cube(uv0[b], neighbours, dist[b], q_in_set[b])
            out[b] = self._apply_delta(states0c[b], uv0[b], p_start_c[b:b + 1],
                                       p_stop_c[b:b + 1], duv, dyaw, yaw0_w[b])
        return out, {"top1_dist": top1_dist, "knn_disagreement": disagreement}

    def predict_occ(self, states0: torch.Tensor, p_start: torch.Tensor,
                    p_stop: torch.Tensor) -> torch.Tensor:
        """Rasterises with `simple_mpc.adapters.occ_from_particles` (the
        SAME rasteriser every OCC_ADAPTERS model and the truth are scored
        with) -- lazily imported so the rest of this package stays usable
        without pulling in `simple_mpc`'s heavier import chain."""
        from simple_mpc.adapters import occ_from_particles
        if self.aggregation not in ("occ_mean", "occ_weighted_mean"):
            return occ_from_particles(self.predict_particles(states0, p_start, p_stop))

        B, n, _ = states0.shape
        states0c = states0.cpu()
        p_start_c, p_stop_c = p_start.cpu().float(), p_stop.cpu().float()
        idx, dist, uv0, q_valid, q_in_set = self._search(states0c, p_start_c, p_stop_c)
        yaw0_w = yaw_from_quat(states0c[..., 3:7])
        occs = torch.zeros(B, 64, 64)
        for b in range(B):
            if not bool(q_valid[b]):
                occs[b] = occ_from_particles(states0c[b:b + 1])[0]
                continue
            cand_states = []
            for j in idx[b]:
                duv, dyaw = self._transfer_one(uv0[b], int(j), q_in_set[b])
                cand_states.append(self._apply_delta(states0c[b], uv0[b], p_start_c[b:b + 1],
                                                      p_stop_c[b:b + 1], duv, dyaw, yaw0_w[b]))
            cand_states = torch.stack(cand_states)          # (k, n, 7)
            cand_occ = occ_from_particles(cand_states)       # (k, 64, 64)
            if self.aggregation == "occ_weighted_mean":
                w = torch.softmax(-dist[b] / max(self.weight_temperature, 1e-9), dim=0)
                occs[b] = (cand_occ * w.view(-1, 1, 1)).sum(dim=0)
            else:
                occs[b] = cand_occ.mean(dim=0)
        return occs


class NearestTransitionPredictor(RetrievalPredictor):
    """Backward-compatible alias for the v0 naive 1-NN predictor: k=1,
    `aggregation="nn1"`, plain (uncapped, unweighted, mean) chamfer distance
    over the given window -- exactly the v0 behaviour, now implemented on
    top of the shared, configurable `RetrievalPredictor` / `distance.py`."""

    def __init__(self, bank: TransitionBank, window_u=None, window_v=None, bank_chunk: int = 1500):
        cfg = DistanceConfig()
        if window_u is not None:
            cfg.window_u = window_u
        if window_v is not None:
            cfg.window_v = window_v
        super().__init__(bank, cfg=cfg, k=1, aggregation="nn1", bank_search_chunk=bank_chunk)
        self.name = "retrieval_1nn"
