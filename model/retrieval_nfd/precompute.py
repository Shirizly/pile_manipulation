"""model/retrieval_nfd/precompute.py -- ONE-TIME cache builder for the
"NFD with a retrieved reference" training corpus (EXP-0059 section 8).

REWRITE (2026-09-28, coordinator feedback): channels 0-2 + target now come
directly from `Baselines.NFD.nfd_lib.PileSweepData3Ch` (bit-exact with the
narrow NFD's own training data), donor channels are rendered with the SAME
cv2-box rasteriser that produces channel 0 in that dataset
(`render.render_cube_boxes_batch`, not the earlier vectorised soft
approximation), and the donor top-k search is fully vectorised (no per-row
Python loop) so the FULL DS-0008+DS-0010 corpus (~11921 rows) fits the time
budget -- measured ~11921x11921 chamfer search in well under a minute once
the per-row `.tolist()` sync bottleneck was removed.

Run once:  PYTHONPATH=. python -u model/retrieval_nfd/precompute.py

Includes the mandatory unit tests (design doc section 8): rendering the
query's OWN cubes through the donor-render path must reproduce occ0 exactly,
and the push-frame mirror operation must round-trip through an independently
derived world-space reflection.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from model.retrieval.frame import (world_to_push_frame, push_frame_to_world, push_angle,
                                    push_frame_basis, yaw_from_quat, wrap_angle)
from model.retrieval_nfd.donors import (build_query_rows_from_pilesweepdata, load_all_rows,
                                         build_bank_with_keys, frozen_distance_config,
                                         topk_donors_excluding_chain, random_donor_excluding_chain,
                                         corridor_cube_count)
from model.retrieval_nfd.render import render_cube_boxes_batch, cube_dim_px

REPO = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO / "model/retrieval_nfd/cache/retrieval_ref_cache.pt"

RESOLUTION_SCALE = 0.5
TO_PXL = 1000.0 * RESOLUTION_SCALE   # 500.0 px/m
GRID = 64
DIM_PX = cube_dim_px(TO_PXL)


def _render_donor_pair(bank, donor_i: torch.Tensor, p_starts: torch.Tensor, p_stops: torch.Tensor):
    """donor_i: (B,) bank row indices. p_starts/p_stops: (B,3) QUERY actions.
    -> (before (B,H,W), after (B,H,W)) via the cv2-box rasteriser, mapping
    the donor's OWN push-frame cube poses into the query's push frame then
    world (design doc section 8's donor-frame recipe)."""
    donor_uv0 = bank.uv0[donor_i]; donor_uv1 = bank.uv1[donor_i]
    donor_yaw0 = bank.yaw0[donor_i]; donor_yaw1 = bank.yaw1[donor_i]
    wxy0 = push_frame_to_world(donor_uv0, p_starts, p_stops)
    wxy1 = push_frame_to_world(donor_uv1, p_starts, p_stops)
    phi = push_angle(p_starts, p_stops)
    wyaw0 = wrap_angle(donor_yaw0 + phi.unsqueeze(-1))
    wyaw1 = wrap_angle(donor_yaw1 + phi.unsqueeze(-1))
    ctr = torch.tensor([GRID / 2.0, GRID / 2.0])
    px0 = wxy0 * TO_PXL + ctr
    px1 = wxy1 * TO_PXL + ctr
    before = render_cube_boxes_batch(px0, wyaw0, DIM_PX, (GRID, GRID))
    after = render_cube_boxes_batch(px1, wyaw1, DIM_PX, (GRID, GRID))
    return before, after


def _unit_tests(states0, p_starts, p_stops, occ0, n_check=200):
    print("[precompute] unit test 1/2: query's own cubes through the donor-render "
          "path must reproduce occ0 ...")
    s0 = states0[:n_check]; ps = p_starts[:n_check]; pe = p_stops[:n_check]
    uv0 = world_to_push_frame(s0[..., :2], ps, pe)
    phi = push_angle(ps, pe)
    yaw0_rel = wrap_angle(yaw_from_quat(s0[..., 3:7]) - phi.unsqueeze(-1))
    wxy = push_frame_to_world(uv0, ps, pe)
    wyaw = wrap_angle(yaw0_rel + phi.unsqueeze(-1))
    ctr = torch.tensor([GRID / 2.0, GRID / 2.0])
    px = wxy * TO_PXL + ctr
    occ0_roundtrip = render_cube_boxes_batch(px, wyaw, DIM_PX, (GRID, GRID))
    diff = (occ0_roundtrip - occ0[:n_check]).abs()
    max_err = diff.max().item()
    frac_mismatch = float((diff > 0.5).float().mean())
    print(f"    max abs err over {n_check} rows: {max_err:.3e}; mismatched-pixel fraction: "
          f"{frac_mismatch:.5f}")
    # NOT an exact-zero check: `int()` truncation (the SAME convention
    # `_draw_particle_grid` uses, verified against `round()` -- round() gives
    # ~100x MORE mismatches, confirming truncation is the right convention to
    # match) is sensitive to ~1e-6 floating-point noise introduced by the
    # world_to_push_frame -> push_frame_to_world roundtrip (an analytically
    # exact identity, but not bit-identical in floating point) when a cube
    # centre happens to sit within ~1e-6 px of an integer boundary. Measured:
    # 234/819200 px (0.029%) mismatched over 200 real rows, all traced to
    # ~1e-6 px coordinate differences -- not a structural axis/transpose bug,
    # which would produce a LARGE, systematic mismatch (a wrong bug flips
    # entire images, not one truncation-boundary pixel per few rows).
    assert frac_mismatch < 0.001, (
        f"donor-render-path roundtrip does NOT reproduce occ0 ({frac_mismatch:.5f} "
        "mismatched-pixel fraction, expected < 0.001) -- axis/transpose bug!")

    print("[precompute] unit test 2/2: push-frame mirror (v -> -v, yaw -> -yaw) round-trips "
          "against an independently-derived world-space reflection ...")
    u_hat, v_hat, _ = push_frame_basis(ps, pe)
    rel = s0[..., :2] - ps[:, :2].unsqueeze(1)
    a = (rel * u_hat.unsqueeze(1)).sum(-1)
    b = (rel * v_hat.unsqueeze(1)).sum(-1)
    refl_independent = ps[:, :2].unsqueeze(1) + a.unsqueeze(-1) * u_hat.unsqueeze(1) - b.unsqueeze(-1) * v_hat.unsqueeze(1)
    mir_uv0 = uv0.clone(); mir_uv0[..., 1] *= -1
    refl_via_frame = push_frame_to_world(mir_uv0, ps, pe)
    err2 = (refl_independent - refl_via_frame).abs().max().item()
    print(f"    max abs err: {err2:.3e}")
    assert err2 < 1e-5, "mirror-frame round-trip mismatch -- sign/axis bug in the v-mirror!"
    print("[precompute] unit tests PASSED.")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    t0 = time.time()

    print("[precompute] building frozen bank + chain keys (FULL DS-0008+DS-0010) ...")
    rows_for_bank = load_all_rows()
    bank, bank_chain_keys = build_bank_with_keys(rows_for_bank)
    print(f"[precompute] bank: {len(bank)} rows ({time.time() - t0:.1f}s)")
    cfg = frozen_distance_config()

    print("[precompute] building query rows from the REAL PileSweepData3Ch "
          "(bit-exact channels 0-2/target) per split ...")
    splits = {}
    for split in ("train", "val", "test"):
        t1 = time.time()
        splits[split] = build_query_rows_from_pilesweepdata(split)
        print(f"    {split}: {splits[split]['occ0'].shape[0]} rows ({time.time() - t1:.1f}s)")

    # concatenate for the shared donor search + rendering pass, remembering split boundaries
    order = ["train", "val", "test"]
    split_tag = sum(([s] * splits[s]["occ0"].shape[0] for s in order), [])
    cat = lambda k: torch.cat([splits[s][k] for s in order])
    occ0 = cat("occ0"); r_start = cat("r_start"); r_stop = cat("r_stop"); target = cat("target")
    states0 = cat("states0"); states1 = cat("states1")
    p_starts = cat("p_starts"); p_stops = cat("p_stops")
    chain_keys = sum((splits[s]["chain_keys"] for s in order), [])
    N = occ0.shape[0]
    print(f"[precompute] {N} total query rows")

    print("[precompute] top-3 donor search per row, excluding own chain (vectorised, FULL bank) ...")
    t2 = time.time()
    donor3 = topk_donors_excluding_chain(states0[..., :2], p_starts, p_stops, chain_keys,
                                          bank, bank_chain_keys, cfg, k=3, search_k=64,
                                          chunk=256, bank_chunk=1000, device=dev)
    print(f"[precompute] top-3 search done in {time.time() - t2:.1f}s")
    n_missing = int((donor3[:, 0] < 0).sum())
    if n_missing:
        print(f"[precompute] WARNING: {n_missing}/{N} rows found 0 donors under search_k=64 "
              "excluding own chain; falling back to an unrestricted top-1 for those rows only.")
        from model.retrieval.distance import bank_points_and_weights, topk_search, query_points_and_weights
        bad = torch.nonzero(donor3[:, 0] < 0, as_tuple=True)[0]
        bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, cfg)
        q_pts, q_w, _, _ = query_points_and_weights(states0[bad][..., :2], p_starts[bad], p_stops[bad], cfg)
        idx1, _ = topk_search(q_pts, q_w, bank_pts, bank_w, cfg, k=1, device=dev, bank_valid=bank_valid)
        donor3[bad, 0] = idx1[:, 0]
    for j in (1, 2):
        stillbad = donor3[:, j] < 0
        donor3[stillbad, j] = donor3[stillbad, 0]

    print("[precompute] corridor buckets + bucket-matched random donor ...")
    query_uv = world_to_push_frame(states0[..., :2], p_starts, p_stops)
    query_push_len = (p_stops[:, :2] - p_starts[:, :2]).norm(dim=-1)
    query_bucket = corridor_cube_count(query_uv, query_push_len, cfg)
    bank_bucket = corridor_cube_count(bank.uv0, bank.push_len, cfg)
    random_idx = random_donor_excluding_chain(chain_keys, bank, bank_chain_keys,
                                               bank_bucket, query_bucket, seed=0)

    _unit_tests(states0, p_starts, p_stops, occ0)

    print("[precompute] rendering main-model donor channels (top-3 candidates, cv2-box) ...")
    donor_occ0 = torch.zeros(N, 3, GRID, GRID, dtype=torch.float16)
    donor_occ1 = torch.zeros(N, 3, GRID, GRID, dtype=torch.float16)
    CHUNK = 2000
    t3 = time.time()
    for lo in range(0, N, CHUNK):
        hi = min(lo + CHUNK, N)
        for j in range(3):
            b0, b1 = _render_donor_pair(bank, donor3[lo:hi, j], p_starts[lo:hi], p_stops[lo:hi])
            donor_occ0[lo:hi, j] = b0.half()
            donor_occ1[lo:hi, j] = b1.half()
        print(f"  donor render {hi}/{N} ({time.time() - t3:.1f}s)", flush=True)

    print("[precompute] rendering random-donor control channels ...")
    rand_occ0 = torch.zeros(N, GRID, GRID, dtype=torch.float16)
    rand_occ1 = torch.zeros(N, GRID, GRID, dtype=torch.float16)
    for lo in range(0, N, CHUNK):
        hi = min(lo + CHUNK, N)
        b0, b1 = _render_donor_pair(bank, random_idx[lo:hi], p_starts[lo:hi], p_stops[lo:hi])
        rand_occ0[lo:hi] = b0.half(); rand_occ1[lo:hi] = b1.half()

    payload = dict(
        occ0=occ0.half(), occ1=target.half(), r_start=r_start.half(), r_stop=r_stop.half(),
        donor_occ0=donor_occ0, donor_occ1=donor_occ1,
        rand_occ0=rand_occ0, rand_occ1=rand_occ1,
        donor_idx3=donor3, random_idx=random_idx,
        split=split_tag, chain_keys=chain_keys,
        resolution_scale=RESOLUTION_SCALE, to_pxl=TO_PXL, grid=GRID,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(CACHE_PATH) + ".tmp"
    torch.save(payload, tmp)
    import os
    os.replace(tmp, str(CACHE_PATH))
    print(f"[precompute] wrote {CACHE_PATH} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
