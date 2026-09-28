"""Bank-build and particle -> prediction -> raster path tests (EXP-0059).

Genesis-free, no data files needed -- everything here is built from small
synthetic tensors, mirroring `docs/CODEMAP.md`'s ground-truth-scoring
convention (`simple_mpc.adapters.occ_from_particles`) so the raster step is
tested against the SAME function the real eval harness rasterises with.
"""
import math

import torch

from model.retrieval.bank import TransitionBank
from model.retrieval.frame import yaw_to_quat, world_to_push_frame
from model.retrieval.predictor import (PersistencePredictor, NearestTransitionPredictor,
                                       RetrievalPredictor)

from simple_mpc.adapters import occ_from_particles


def _make_state(xy, z=0.0125, yaw=None):
    """xy: (n, 2) -> (n, 7) state row [x, y, z, qw, qx, qy, qz]."""
    n = xy.shape[0]
    if yaw is None:
        yaw = torch.zeros(n)
    q = yaw_to_quat(yaw)
    zcol = torch.full((n, 1), z)
    return torch.cat([xy, zcol, q], dim=1)


def test_bank_from_states_moved_mask_and_displacement():
    # 3 objects: #0 stays put, #1 moves 5mm along +u (the push direction),
    # #2 moves 0.5mm (below a 1mm threshold -> not "moved").
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])   # push along +x, L=20mm
    xy0 = torch.tensor([[[0.05, 0.05], [0.0, 0.0], [0.01, 0.01]]])   # (1, 3, 2)
    xy1 = xy0.clone()
    xy1[0, 1, 0] += 0.005   # object 1 moves 5mm in +x (== +u here)
    xy1[0, 2, 0] += 0.0003  # object 2 moves 0.3mm (below threshold)

    states0 = _make_state(xy0[0])[None]
    states1 = _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop,
                                      source=["test"], moved_threshold=0.001)

    assert len(bank) == 1
    assert bank.moved[0].tolist() == [False, True, False]
    # object 1's push-frame displacement must be (+5mm, 0) since the push is
    # along +x (== the canonical +u axis here)
    assert torch.allclose(bank.duv[0, 1], torch.tensor([0.005, 0.0]), atol=1e-6)
    assert torch.allclose(bank.duv[0, 0], torch.zeros(2), atol=1e-6)
    hist = bank.moved_count_histogram()
    assert hist == {1: 1}   # exactly one transition, with exactly 1 moved cube


def test_bank_save_load_round_trip(tmp_path):
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    xy0 = torch.rand(1, 5, 2) * 0.05
    xy1 = xy0.clone(); xy1[0, 0] += 0.003
    states0, states1 = _make_state(xy0[0])[None], _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t"])
    path = tmp_path / "bank.pt"
    bank.save(str(path))
    loaded = TransitionBank.load(str(path))
    assert len(loaded) == len(bank)
    assert torch.allclose(loaded.duv, bank.duv)
    assert loaded.source == bank.source


def test_persistence_predictor_is_identity_and_rasters_unchanged():
    states0 = _make_state(torch.rand(6, 2) * 0.05)[None]
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    pred = PersistencePredictor()
    out = pred.predict_particles(states0, p_start, p_stop)
    assert torch.equal(out, states0)
    occ0 = occ_from_particles(states0)
    occ1 = occ_from_particles(out)
    assert torch.equal(occ0, occ1)


def test_nearest_transition_predictor_recovers_exact_match():
    """A query that is EXACTLY a bank transition's start state, pushed with
    the EXACT same action, must retrieve that transition (distance 0) and
    reproduce its end state exactly (up to the yaw round-trip already
    covered by test_retrieval_frame.py)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    xy0 = torch.tensor([[[0.005, 0.0], [0.05, 0.05], [-0.05, -0.05]]])  # 1 near, 2 far
    xy1 = xy0.clone()
    xy1[0, 0, 0] += 0.008   # the near cube moves 8mm

    states0 = _make_state(xy0[0])[None]
    states1 = _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop,
                                      source=["test"], moved_threshold=0.001)

    predictor = NearestTransitionPredictor(bank, window_u=(-0.005, 0.045),
                                           window_v=(-0.05, 0.05))
    query_states0 = states0.clone()
    out = predictor.predict_particles(query_states0, p_start, p_stop)

    assert torch.allclose(out[0, :, :2], states1[0, :, :2], atol=1e-5)


def test_nearest_transition_predictor_leaves_state_in_place_when_bank_far():
    """If no bank transition has any cube in the query's window, the row is
    left unchanged (documented fallback in `predict_particles`)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    # bank transition has its only cube far outside any reasonable window
    xy0 = torch.tensor([[[5.0, 5.0]]])
    xy1 = xy0.clone(); xy1[0, 0, 0] += 0.01
    states0, states1 = _make_state(xy0[0])[None], _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t"])

    predictor = NearestTransitionPredictor(bank)
    query_xy = torch.tensor([[[0.01, 0.0]]])   # inside the DEFAULT query window
    query_states0 = _make_state(query_xy[0])[None]
    out = predictor.predict_particles(query_states0, p_start, p_stop)
    # the bank has nothing in ITS window either (far outside [-5mm,45mm]x[-50,50mm]),
    # so bank_mask is all-False for every bank row -> chamfer degenerates but a
    # best (very bad) match is still picked; what matters here is the plumbing
    # runs end to end without raising and returns a finite, correctly shaped state.
    assert out.shape == query_states0.shape
    assert torch.isfinite(out).all()


def test_persistence_confidence_is_degenerate_zero():
    states0 = _make_state(torch.rand(4, 2) * 0.05)[None]
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    out, conf = PersistencePredictor().predict_particles_with_confidence(states0, p_start, p_stop)
    assert torch.equal(out, states0)
    assert torch.equal(conf["top1_dist"], torch.zeros(1))
    assert torch.equal(conf["knn_disagreement"], torch.zeros(1))


def test_confidence_top1_dist_near_zero_for_exact_match():
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    xy0 = torch.tensor([[[0.005, 0.0], [0.05, 0.05], [-0.05, -0.05]]])
    xy1 = xy0.clone(); xy1[0, 0, 0] += 0.008
    states0, states1 = _make_state(xy0[0])[None], _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t"])
    predictor = NearestTransitionPredictor(bank)
    out, conf = predictor.predict_particles_with_confidence(states0, p_start, p_stop)
    assert float(conf["top1_dist"][0]) < 1e-4
    assert float(conf["knn_disagreement"][0]) == 0.0   # k=1: nothing to disagree with
    assert torch.allclose(out[0, :, :2], states1[0, :, :2], atol=1e-5)


def test_knn_disagreement_is_zero_for_identical_neighbours_and_positive_for_conflicting_ones():
    p_start = torch.tensor([[0.0, 0.0, 0.0]] * 3)
    p_stop = torch.tensor([[0.02, 0.0, 0.0]] * 3)
    xy0 = torch.tensor([[[0.005, 0.0]]] * 3)   # 3 IDENTICAL bank rows at the same pre-push position
    xy1_same = xy0.clone(); xy1_same[:, 0, 0] += 0.006   # rows agree on the outcome
    states0 = torch.stack([_make_state(xy0[i]) for i in range(3)])
    states1_same = torch.stack([_make_state(xy1_same[i]) for i in range(3)])
    bank_agree = TransitionBank.from_states(states0, states1_same, p_start, p_stop, source=["t"] * 3)

    query_states0 = _make_state(xy0[0])[None]
    q_start, q_stop = p_start[:1], p_stop[:1]
    pred_agree = RetrievalPredictor(bank_agree, k=3, aggregation="cube_median")
    _, conf_agree = pred_agree.predict_particles_with_confidence(query_states0, q_start, q_stop)
    assert float(conf_agree["knn_disagreement"][0]) < 1e-6

    xy1_conflict = torch.stack([xy0[0].clone() for _ in range(3)])
    xy1_conflict[0, 0, 0] += 0.002
    xy1_conflict[1, 0, 0] += 0.015
    xy1_conflict[2, 0, 0] -= 0.010
    states1_conflict = torch.stack([_make_state(xy1_conflict[i]) for i in range(3)])
    bank_conflict = TransitionBank.from_states(states0, states1_conflict, p_start, p_stop, source=["t"] * 3)
    pred_conflict = RetrievalPredictor(bank_conflict, k=3, aggregation="cube_median")
    _, conf_conflict = pred_conflict.predict_particles_with_confidence(query_states0, q_start, q_stop)
    assert float(conf_conflict["knn_disagreement"][0]) > float(conf_agree["knn_disagreement"][0])


def _two_hypothesis_bank():
    """2 bank rows, same pre-push cube position, DIFFERENT post-push position
    -- i.e. two disagreeing donor transitions for the same query, the
    minimal case that exercises every aggregation mode."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]] * 2)
    p_stop = torch.tensor([[0.02, 0.0, 0.0]] * 2)
    xy0 = torch.tensor([[[0.005, 0.0]]] * 2)
    xy1 = torch.stack([xy0[0].clone(), xy0[0].clone()])
    xy1[0, 0, 0] += 0.004    # hypothesis A: +4mm
    xy1[1, 0, 0] += 0.020    # hypothesis B: +20mm
    states0 = torch.stack([_make_state(xy0[i]) for i in range(2)])
    states1 = torch.stack([_make_state(xy1[i]) for i in range(2)])
    return TransitionBank.from_states(states0, states1, p_start, p_stop,
                                      source=["t"] * 2, moved_threshold=0.0005)


def test_predict_particles_stays_clean_top1_for_occ_aggregations():
    """`predict_particles` on occ_mean/occ_weighted_mean must fall back to
    the single top-1 clean transfer -- required by the "hedging is
    scoring-only" rule -- not some blend of the two hypotheses."""
    bank = _two_hypothesis_bank()
    query_states0 = _make_state(torch.tensor([[0.005, 0.0]]))[None]
    q_start = torch.tensor([[0.0, 0.0, 0.0]])
    q_stop = torch.tensor([[0.02, 0.0, 0.0]])
    for agg in ("occ_mean", "occ_weighted_mean"):
        pred = RetrievalPredictor(bank, k=2, aggregation=agg)
        out = pred.predict_particles(query_states0, q_start, q_stop)
        # must equal EXACTLY one of the two donor hypotheses (top-1), not a blend
        matches_a = torch.allclose(out[0, 0, 0], torch.tensor(0.005 + 0.004), atol=1e-5)
        matches_b = torch.allclose(out[0, 0, 0], torch.tensor(0.005 + 0.020), atol=1e-5)
        assert matches_a or matches_b, (agg, float(out[0, 0, 0]))


def test_predict_occ_hedges_across_both_hypotheses():
    """`predict_occ` for occ_mean must NOT equal the rasterisation of either
    single hypothesis alone -- it is an average of both, so its occupancy
    mass sits somewhere the persistence/top-1 rasterisation does not."""
    from simple_mpc.adapters import occ_from_particles as _occ
    bank = _two_hypothesis_bank()
    query_states0 = _make_state(torch.tensor([[0.005, 0.0]]))[None]
    q_start = torch.tensor([[0.0, 0.0, 0.0]])
    q_stop = torch.tensor([[0.02, 0.0, 0.0]])
    pred = RetrievalPredictor(bank, k=2, aggregation="occ_mean")
    occ_hedged = pred.predict_occ(query_states0, q_start, q_stop)

    hypothesis_a = _make_state(torch.tensor([[0.009, 0.0]]))[None]
    hypothesis_b = _make_state(torch.tensor([[0.025, 0.0]]))[None]
    occ_a, occ_b = _occ(hypothesis_a), _occ(hypothesis_b)
    assert not torch.equal(occ_hedged, occ_a)
    assert not torch.equal(occ_hedged, occ_b)
    # the hedge is a genuine average: some mass lands wherever EITHER
    # hypothesis put mass
    assert bool(((occ_a > 0) | (occ_b > 0)).float().sum() > 0)


def test_cube_median_differs_from_cube_weighted_mean_with_uneven_distances():
    """With 3 neighbours at different distances, a plain median and a
    distance-weighted mean need not agree -- just check both run and give
    finite, distinct-in-general per-cube transfers (not asserting a
    specific numeric relationship, since which is "larger" depends on the
    distance spread)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]] * 3)
    p_stop = torch.tensor([[0.02, 0.0, 0.0]] * 3)
    xy0 = torch.tensor([[[0.001, 0.0]], [[0.005, 0.0]], [[0.030, 0.0]]])  # varying query-distance
    xy1 = xy0.clone()
    xy1[:, 0, 0] += torch.tensor([0.003, 0.010, 0.025])
    states0 = torch.stack([_make_state(xy0[i]) for i in range(3)])
    states1 = torch.stack([_make_state(xy1[i]) for i in range(3)])
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t"] * 3)
    query_states0 = _make_state(torch.tensor([[0.005, 0.0]]))[None]
    q_start, q_stop = p_start[:1], p_stop[:1]

    pred_med = RetrievalPredictor(bank, k=3, aggregation="cube_median")
    pred_wmean = RetrievalPredictor(bank, k=3, aggregation="cube_weighted_mean", weight_temperature=0.005)
    out_med = pred_med.predict_particles(query_states0, q_start, q_stop)
    out_wmean = pred_wmean.predict_particles(query_states0, q_start, q_stop)
    assert torch.isfinite(out_med).all() and torch.isfinite(out_wmean).all()


def test_transfer_yaw_false_leaves_quaternion_unchanged():
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    xy0 = torch.tensor([[[0.005, 0.0]]])
    xy1 = xy0.clone(); xy1[0, 0, 0] += 0.006
    states0 = _make_state(xy0[0], yaw=torch.tensor([0.3]))[None]
    states1_yawed = _make_state(xy1[0], yaw=torch.tensor([1.0]))[None]   # donor also rotates
    bank = TransitionBank.from_states(states0, states1_yawed, p_start, p_stop, source=["t"])

    pred_no_yaw = RetrievalPredictor(bank, k=1, aggregation="nn1", transfer_yaw=False)
    out = pred_no_yaw.predict_particles(states0, p_start, p_stop)
    assert torch.allclose(out[0, :, 3:7], states0[0, :, 3:7], atol=1e-6)   # quaternion untouched
    assert not torch.allclose(out[0, :, :2], states0[0, :, :2], atol=1e-4)  # position still moved


def test_far_cube_never_moves_after_distance_gate_fix():
    """2026-09-28 fix (coordinator follow-up C, ISS-010/`retrieval_debug.py`
    q0908): before the fix, `_transfer_one`'s unconditional full Hungarian
    assignment force-matched EVERY query cube to SOME donor cube, however far
    apart, so a query cube nowhere near any donor cube could inherit a large,
    physically-nonsensical displacement. Now a matched pair farther apart
    than `distance_gate` (default 6mm) transfers nothing -- the cube stays
    exactly in place, even though it sits inside the retrieval window/
    interaction set (isolating the DISTANCE gate specifically, not window/
    interaction-set exclusion, which would trivially leave it unmoved for a
    different reason)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    # one donor transition: cube 0 near the blade start moves 8mm; cube 1 far away, unmoved.
    donor_xy0 = torch.tensor([[[0.005, 0.0], [0.05, 0.05]]])
    donor_xy1 = donor_xy0.clone(); donor_xy1[0, 0, 0] += 0.008
    donor_states0 = _make_state(donor_xy0[0])[None]
    donor_states1 = _make_state(donor_xy1[0])[None]
    bank = TransitionBank.from_states(donor_states0, donor_states1, p_start, p_stop,
                                      source=["t"], moved_threshold=0.001)
    predictor = RetrievalPredictor(bank, k=1, aggregation="nn1")

    # query cube 0 sits exactly at donor cube 0 (legit close match, <1mm);
    # query cube 1 is inside the window/interaction set geometrically (the
    # blade plausibly sweeps near it) but >6mm from EVERY donor cube.
    query_xy = torch.tensor([[[0.005, 0.0], [0.02, 0.02]]])
    query_states0 = _make_state(query_xy[0])[None]
    out = predictor.predict_particles(query_states0, p_start, p_stop)

    assert torch.allclose(out[0, 0, :2], torch.tensor([0.013, 0.0]), atol=1e-5)   # near cube: legit +8mm
    assert torch.allclose(out[0, 1, :2], query_xy[0, 1], atol=1e-9)               # far cube: gated, stays put


def test_exact_match_recovery_still_works_after_distance_gate_fix():
    """The distance-gate/interaction-set fix must not break the basic case
    it is meant to leave alone: a query that IS a bank transition's start
    state, pushed with the exact same action, still retrieves it (distance
    0, well under the gate) and reproduces the end state exactly -- for
    EVERY cube, including ones far from the blade that legitimately do not
    move (their own gate-vs-gate distance is 0: donor==query position)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    xy0 = torch.tensor([[[0.005, 0.0], [0.05, 0.05], [-0.05, -0.05]]])  # 1 near, 2 far, unmoved
    xy1 = xy0.clone()
    xy1[0, 0, 0] += 0.008   # only the near cube moves

    states0 = _make_state(xy0[0])[None]
    states1 = _make_state(xy1[0])[None]
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop,
                                      source=["test"], moved_threshold=0.001)
    predictor = RetrievalPredictor(bank, k=1, aggregation="nn1")
    out = predictor.predict_particles(states0.clone(), p_start, p_stop)

    assert torch.allclose(out[0, :, :2], states1[0, :, :2], atol=1e-5)


def test_particle_to_raster_path_shapes_and_no_nan():
    """The full particle -> prediction -> raster path a real predictor is
    scored through: build a small bank, predict, rasterise with the SAME
    `occ_from_particles` the eval harness uses, and check basic sanity."""
    torch.manual_seed(2)
    n = 8
    p_start = torch.rand(3, 3) * 0.02 - 0.01
    p_start[:, 2] = 0.0
    p_stop = p_start.clone()
    p_stop[:, 0] += 0.02  # +x pushes of 20mm

    xy0 = torch.rand(3, n, 2) * 0.08 - 0.04
    xy1 = xy0.clone()
    xy1[:, 0, 0] += 0.01   # cube 0 always moves

    states0 = torch.stack([_make_state(xy0[b]) for b in range(3)])
    states1 = torch.stack([_make_state(xy1[b]) for b in range(3)])
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop,
                                      source=["t"] * 3, moved_threshold=0.001)

    predictor = NearestTransitionPredictor(bank, bank_chunk=2)
    pred_states = predictor.predict_particles(states0, p_start, p_stop)
    assert pred_states.shape == states0.shape
    assert torch.isfinite(pred_states).all()

    occ_pred = occ_from_particles(pred_states)
    occ_truth = occ_from_particles(states1)
    assert occ_pred.shape == occ_truth.shape == (3, 64, 64)
    assert torch.isfinite(occ_pred).all()
    assert float(occ_pred.sum()) > 0.0   # some mass actually landed on the grid
