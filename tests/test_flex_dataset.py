"""FlexData loader (EXP-0061): shapes, axis/frame convention, plate geometry,
split disjointness, DS-0020 v2 leakage exclusion + chaining. Genesis-free.
Synthetic-cache tests always run; the real-corpus tests skip if the
caches/splits have not been built (``python -u -m FlexData.build_cache
ds0020|splits|paths``, ``python -u -m FlexData.image_mask ds0020``). DS-0020 v1
(archived, ``old_data/_ported_v1``) tests skip if its cache is absent."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from FlexData.dataset import FlexPileData, flex_xz_to_table, rasterize_disk  # noqa: E402

DS20 = REPO / "datasets/DS-0020-training-data-flex-N864"
DS20_V1 = DS20 / "old_data/_ported_v1"
DS19 = REPO / "datasets/DS-0019-slates-flex-pile-varN"


def _synthetic_instance(tmp_path, xz_states, actions):
    """One-trajectory cache in the DS-0020 chunk format + an instance cfg."""
    cache = tmp_path / "cache"
    cache.mkdir()
    xz = np.asarray(xz_states, np.float16)[None]                  # (1, K+1, P, 2) raw flex (x, z)
    acts = np.asarray(actions, np.float32)[None]                  # (1, K, 4)
    md = np.full(acts.shape[:2], 5.0, np.float32)
    np.savez(cache / "chunk_000.npz", xz=xz, actions=acts, traj_ids=np.array([0], np.int32), max_disp=md)
    (tmp_path / "splits.json").write_text(json.dumps({"splits": {"train": [0], "val": [], "test": []}}))
    return dict(
        id="SYN", kind="trajectories", cache_dir=str(cache), split_file=str(tmp_path / "splits.json"),
        grid=dict(half_extent=7.2, resolution=64, footprint_radius_px=1.0),
        plate=dict(width=2.4, thickness=0.225, height=None, sigma_px=0.75),
        flags=dict(null_disp=0.1, escape_abs=10.0, max_out_of_grid_frac=0.01),
    )


def test_rasterize_disk_matches_bruteforce():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 66, size=(300, 2)).astype(np.float32)
    for r in (0.6, 1.0, 1.7):
        got = rasterize_disk(pts, 64, 64, r)
        ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        d2 = (ii[None] - pts[:, 0, None, None]) ** 2 + (jj[None] - pts[:, 1, None, None]) ** 2
        want = (d2 <= r * r).any(0).astype(np.float32)
        assert np.array_equal(got, want)


def test_axis_and_frame_convention(tmp_path):
    """row = X = x_flex, col = Y = -z_flex; pixel = world * to_pxl + 32."""
    x, z = 3.0, 2.0                         # one particle, flex coords
    st = [[[x, z]], [[x, z]]]
    act = [[0.0, 0.0, 1.0, 0.0]]            # push along +X
    ds = FlexPileData(_synthetic_instance(tmp_path, st, act), "train", exclude_flagged=False, verbose=False)
    (inp, _), tgt = ds[0]
    assert inp.shape == (3, 64, 64) and tgt.shape == (64, 64)
    r, c = np.unravel_index(int(inp[0].argmax()), (64, 64))
    to_pxl = 64 / 14.4
    assert abs(r - (x * to_pxl + 32)) <= 1 and abs(c - (-z * to_pxl + 32)) <= 1
    assert np.allclose(flex_xz_to_table(np.array([[x, z]])), [[x, -z]])
    # plate channels: centroid at the mapped endpoints, long side perpendicular to the push
    for ch, p in ((1, act[0][:2]), (2, act[0][2:])):
        m = inp[ch]
        w = m / m.sum()
        ii = torch.arange(64.0)
        cr, cc = float((w.sum(1) * ii).sum()), float((w.sum(0) * ii).sum())
        assert abs(cr - (p[0] * to_pxl + 32)) < 0.05 and abs(cc - (p[1] * to_pxl + 32)) < 0.05
    rows = torch.nonzero(inp[1] > 0.3)
    span_r = int(rows[:, 0].max() - rows[:, 0].min()) + 1
    span_c = int(rows[:, 1].max() - rows[:, 1].min()) + 1
    assert span_c > 3 * span_r              # push along X (rows) -> plate spans columns
    assert abs(ds.plate_width_px - 2.4 * to_pxl) < 1e-6
    # PileSweepData-compatible geometry other consumers read
    assert ds.configs[0]["plate"]["size"][0] == 2.4
    assert ds.workspace_bounds == ((-7.2, -7.2), (7.2, 7.2))
    assert max(0.5, 1.5 * ds.resolution_scale) == pytest.approx(0.75)


def test_flags_drop_null_and_escaped(tmp_path):
    base = [[0.0, 0.0], [1.0, 1.0]]
    st = [base, base, [[0.0, 0.0], [50.0, 1.0]]]            # push 0 null, push 1 escapes
    act = [[0, 0, 1, 0], [0, 0, 1, 0]]
    cfg = _synthetic_instance(tmp_path, st, act)
    np.savez(Path(cfg["cache_dir"]) / "chunk_000.npz", xz=np.asarray(st, np.float16)[None],
             actions=np.asarray(act, np.float32)[None], traj_ids=np.array([0], np.int32),
             max_disp=np.array([[0.0, 49.0]], np.float32))
    ds = FlexPileData(cfg, "train", exclude_flagged=True, verbose=False)
    assert bool(ds.flags["null"][0]) and bool(ds.flags["escaped"][1]) and len(ds) == 0


def test_image_mask_hook(tmp_path):
    """occ_source=image_mask reads precomputed masks (input AND target); a
    missing mask file raises instead of silently using the particle raster."""
    st = [[[0.0, 0.0]], [[1.0, 0.0]]]
    cfg = _synthetic_instance(tmp_path, st, [[0.0, 0.0, 1.0, 0.0]])
    with pytest.raises(FileNotFoundError):
        FlexPileData(cfg, "train", exclude_flagged=False, verbose=False, occ_source="image_mask")
    m = np.zeros((1, 2, 64, 64), np.uint8)
    m[0, 0, 5, 7] = 1
    m[0, 1, 9, 11] = 1
    np.savez(Path(cfg["cache_dir"]) / "image_masks.npz", traj_ids=np.array([0]), masks=m)
    ds = FlexPileData(cfg, "train", exclude_flagged=False, verbose=False, occ_source="image_mask")
    (inp, _), tgt = ds[0]
    assert inp[0].sum() == 1 and inp[0, 5, 7] == 1 and tgt.sum() == 1 and tgt[9, 11] == 1
    ds_p = FlexPileData(cfg, "train", exclude_flagged=False, verbose=False)     # default: particles
    assert ds_p.occ_source == "particles" and ds_p[0][0][0][0].sum() > 1


def test_v2_manifest_source_variable_particle_count(tmp_path):
    """cache_format traj_manifest_v2: per-trajectory particle counts differ; rows,
    before/after chaining (after of push k == before of push k+1) and traj ids."""
    cache = tmp_path / "cache"; cache.mkdir()
    rng = np.random.default_rng(0)
    P = [3, 5]
    xz = [rng.uniform(-3, 3, (11, p, 2)).astype(np.float16) for p in P]
    acts = np.tile(np.array([0, 0, 1, 0], np.float32), (2, 10, 1))
    np.savez(cache / "v2_chunk_001.npz", traj_ids=np.array([107, 150], np.int32),
             p_off=np.array([0, 3, 8]), xz=np.concatenate(xz, 1), actions=acts,
             max_disp=np.full((2, 10), 5.0, np.float32), n_particles=np.array(P, np.int32),
             n_rigids=np.array([1, 2], np.int32), init_pos=np.array([0, 1], np.int8),
             push_length=np.ones((2, 10), np.float32))
    (tmp_path / "splits.json").write_text(json.dumps({"splits": {"train": [150], "val": [107], "test": []}}))
    cfg = dict(id="SYN2", kind="trajectories", cache_format="traj_manifest_v2", cache_dir=str(cache),
               split_file=str(tmp_path / "splits.json"),
               grid=dict(half_extent=7.2, resolution=64, footprint_radius_px=1.0),
               plate=dict(width=2.4, thickness=0.225, height=None, sigma_px=0.75),
               flags=dict(null_disp=0.1, escape_abs=10.0, max_out_of_grid_frac=0.01))
    ds = FlexPileData(cfg, "train", exclude_flagged=False, verbose=False)
    assert len(ds) == 10 and {ds.get_run_index(i) for i in range(10)} == {150}
    for k in range(9):
        assert torch.equal(ds.particles_after(k), ds.particles_before(k + 1))
        assert ds.particles_before(k).shape == (5, 2)
    assert np.allclose(ds.particles_before(0).numpy(), flex_xz_to_table(xz[1][0].astype(np.float32)))
    assert len(FlexPileData(cfg, "val", exclude_flagged=False, verbose=False)) == 10


@pytest.mark.skipif(not (DS20_V1 / "splits.json").exists(), reason="DS-0020 v1 archive absent")
def test_ds0020_v1_split_by_trajectory_disjoint():
    sp = json.loads((DS20_V1 / "splits.json").read_text())["splits"]
    tr, va = set(sp["train"]), set(sp["val"])
    assert not tr & va
    assert tr | va == set(range(2000)) and len(va) == 200


@pytest.mark.skipif(not (DS20 / "splits.json").exists() or not (DS20 / "cache/manifest.json").exists(),
                    reason="DS-0020 v2 splits/cache not built")
def test_ds0020_v2_split_disjoint_and_leak_excluded():
    """v2: by trajectory, disjoint, NO trajectory 0-99 anywhere (leakage with DS-0019),
    covers every complete trajectory >= 100, 10 % val."""
    spj = json.loads((DS20 / "splits.json").read_text())
    sp = spj["splits"]
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not tr & va and not te
    assert not (tr | va) & set(range(100))
    assert set(spj["excluded"]["ids"]) == set(range(100))
    from FlexData.build_cache import read_v2_manifest, v2_complete_trajectories
    complete = set(v2_complete_trajectories(read_v2_manifest()))
    assert tr | va == {t for t in complete if t >= 100}
    assert len(va) == round(0.1 * len(tr | va))
    ds = FlexPileData(DS20 / "config.yaml", "val", exclude_flagged=False, verbose=False)
    assert set(ds.src.traj_ids.tolist()) == va


@pytest.mark.skipif(not (DS20 / "cache/manifest.json").exists(), reason="DS-0020 v2 cache not built")
def test_ds0020_v2_chaining_after_k_is_before_k1():
    """Chaining verified from the data: particles moved by push k (k >= 1) lie in
    push k's swept box when its before-state is push k-1's after-state, and do NOT
    when it is the initial state (the reset hypothesis). Also: the cached state 0
    equals initial_particles.npy, and each after-state equals its manifest file."""
    from FlexData.build_cache import DS0020_RAW, read_v2_manifest
    ds = FlexPileData(DS20 / "config.yaml", "all", exclude_flagged=False, verbose=False, occ_source="particles")
    src = ds.src
    vm = read_v2_manifest()
    rng = np.random.default_rng(1)

    def outside(b, a, aft):
        s, e = a[:2], a[2:]
        L = float(np.hypot(*(e - s))); ux = (e - s) / max(L, 1e-9)
        rel = b - s; u, v = rel @ ux, rel @ np.array([-ux[1], ux[0]])
        mv = np.hypot(*(aft - b).T) > 0.25
        return np.nan if mv.sum() == 0 else float((mv & ~((u > -1) & (u < L + 2.5) & (np.abs(v) < 2.5))).sum() / mv.sum())
    chain, reset = [], []
    for t in rng.choice(len(src.traj_ids), 40, replace=False):
        tid = int(src.traj_ids[t])
        raw0 = np.load(DS0020_RAW / vm["inits"][tid]["positions_path"]).reshape(-1, 4)[:, [0, 2]]
        assert np.abs(src.state(t, 0).astype(np.float32) - raw0).max() < 4e-3
        k = int(rng.integers(0, 10))
        rawk = np.load(DS0020_RAW / vm["steps"][tid][k]["after_positions_path"]).reshape(-1, 4)[:, [0, 2]]
        ok = np.isfinite(rawk).all(1) & (np.abs(rawk) < 10).all(1)
        assert np.abs(src.state(t, k + 1).astype(np.float32)[ok] - rawk[ok]).max() < 4e-3
        init = flex_xz_to_table(src.state(t, 0))
        for k in range(1, 10):
            a = src.actions[t, k].astype(np.float64)
            aft = flex_xz_to_table(src.state(t, k + 1))
            chain.append(outside(flex_xz_to_table(src.state(t, k)), a, aft))
            reset.append(outside(init, a, aft))
    assert np.nanmedian(chain) < 0.01 and np.nanmean(chain) < 0.03
    assert np.nanmean(reset) > 0.15


@pytest.mark.skipif(not (DS19 / "splits.json").exists(), reason="DS-0019 splits not built")
def test_ds0019_slates_are_valid_manifest_rows():
    sp = json.loads((DS19 / "splits.json").read_text())
    valid = {(r["state_idx"], r["action_idx"]) for r in map(json.loads, open(DS19 / "manifest.jsonl"))
             if r["type"] == "action" and r["valid"]}
    listed = {(int(s), a) for s, al in sp["slates"].items() for a in al}
    assert listed == valid and len(sp["slates"]) == 100


@pytest.mark.parametrize("cfg_name,cache_dir", [
    ("flex_ds0020_train_ds0019_test.yaml", DS20_V1 / "cache"),       # archived v1 (EXP-0061 weights)
    ("flex_ds0020v2_train_ds0019_test.yaml", DS20 / "cache"),         # current v2, image-mask input
])
def test_ds0020_registry_batch_and_swept_removal(cfg_name, cache_dir):
    """Registered `flex` type emits the Eulerian batch keys; on real rows the
    removed occupancy lies inside the plate-width swept region (frame check:
    with the stored actions read as literal (x, z) this fails)."""
    if not (cache_dir / "manifest.json").exists():
        pytest.skip(f"cache not built: {cache_dir}")
    import yaml
    from fit_linear_foresight import actions_to_pixels, plate_width_px, swept_region_mask
    from registry.dataset_registry import build_dataset
    cfg = yaml.safe_load((REPO / "configs/dataset" / cfg_name).read_text())
    if cfg.get("occ_source") == "image_mask" and not (cache_dir / "image_masks.DONE").exists():
        pytest.skip("image-mask cache not built")
    ds = build_dataset(cfg, "val")
    b = ds[0]
    assert b["input"].shape == (3, 64, 64) and b["target"].shape == (64, 64)
    assert {"input", "target", "current_occupancy", "target_occupancy"} <= set(b)
    raw = ds.raw_dataset
    idx = list(range(0, 400, 20))
    acts = torch.stack([raw.get_raw_action(i) for i in idx])
    s, e = actions_to_pixels(acts, *raw.workspace_bounds, (64, 64))
    p = plate_width_px(raw, 64)
    reg = swept_region_mask(s, e, (64, 64), 0.5 * p + 2.0, 0.5 * p)
    rem = rem_in = 0.0
    for k, i in enumerate(idx):
        (x, _), y = raw[i]
        removed = ((x[0] > 0.5) & (y < 0.5)).float()
        rem += float(removed.sum()); rem_in += float((removed * reg[k]).sum())
    assert rem_in / rem > 0.95


DS21 = REPO / "datasets/DS-0021-flex-carrots-countgroups-train/data"
DS22 = REPO / "datasets/DS-0022-flex-carrots-countgroups-test-slates/data"


def _push_disp_cosines(root: Path, kind: str, n_rows: int = 40):
    """Mean cos(mean displacement of moved, non-escaped particles, push direction) under the
    literal (x, z) and the (x, -z) reading of the stored action (invariant flex-action-frame-neg-z)."""
    recs = [json.loads(l) for l in open(root / "manifest.jsonl")]
    rows = [r for r in recs if r["type"] == kind and r["valid"]]
    rng = np.random.default_rng(0)
    cos = {1: [], -1: []}
    for i in rng.choice(len(rows), n_rows, replace=False):
        r = rows[i]
        k = r.get("step_idx", 0)
        before = root / str(r["state_idx"]) / ("initial_particles.npy" if (kind == "action" or k == 0)
                                               else f"{k - 1}_after_particles.npy")
        p0 = np.load(before).reshape(-1, 4)[:, [0, 2]]
        p1 = np.load(root / r["after_positions_path"]).reshape(-1, 4)[:, [0, 2]]
        d = p1 - p0
        ok = (np.linalg.norm(d, axis=1) > 0.1) & (np.abs(p1).max(1) < 10)
        if ok.sum() < 3:
            continue
        md = d[ok].mean(0)
        a = np.asarray(r["action"], dtype=float)
        for sg in (1, -1):
            v = np.array([a[2] - a[0], sg * (a[3] - a[1])])
            cos[sg].append(float(md @ v / (np.linalg.norm(md) * np.linalg.norm(v))))
    return float(np.mean(cos[1])), float(np.mean(cos[-1]))


@pytest.mark.parametrize("root,kind", [(DS21, "transition"), (DS22, "action")], ids=["DS-0021", "DS-0022"])
def test_flex_countgroup_actions_are_x_negz(root, kind):
    """EXP-0064: the count-group corpora store actions as (x, -z) like DS-0019/20; the literal
    reading must be ~uncorrelated with the observed motion (negated control)."""
    if not (root / "manifest.jsonl").exists():
        pytest.skip(f"payload absent: {root}")
    lit, negz = _push_disp_cosines(root, kind)
    assert negz > 0.9, negz
    assert abs(lit) < 0.3, lit
