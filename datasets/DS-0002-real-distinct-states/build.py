"""Build DS-0002 -- a deduplicated corpus of real, genuinely distinct particle
states, pooled across every existing real corpus in Genesis/data/.

Why: the encoder and value function only need STATES (never transitions), but
every existing corpus is either slate-shaped (one pile x ~1000 near-identical
post-push states) or was never checked for cross-file duplication. Training
either component on near-duplicate rows with a row-random split silently
picks a far-too-small regularisation strength -- this is the
`readout-cv-folds-slate-aware` bug (experiments/INVARIANTS.md), hit four times.

Rule implemented here (see the task brief / DATASET.md for the full rationale):
  - ALL distinct INITIAL states, from every corpus swept.
  - POST-sweep states ONLY from corpora whose transitions are INDEPENDENT
    (overnight_randlen_{train,test}, Sean/*) -- those are genuinely diverse.
  - POST-sweep states from slate pools (slates_binned, slates_multistep) are
    near-duplicates of one pile -- keep at most SLATE_POST_SAMPLE per slate,
    sampled, never wholesale.

Dedup criterion (defensible, explicit): round each particle's (x,y) to
DEDUP_TOL_M, sort the per-row (qx,qy) pairs (permutation-invariant -- particle
storage order is not meaningful), hash the resulting canonical byte string.
Two rows with the same hash (and the same n_objects) are the same state.
Quaternion / z are NOT part of the key: dedup is about pile *layout*, and
z is ~constant (single layer) while yaw differs continuously run to run even
for a physically-identical layout in this simulator's convention.

Every row also carries a `group` key -- coarse but SAFE: one group per
source file (randlen/Sean) or one group per slate (slates_binned/multistep).
Rows in the same group may share physical ancestry (chained pushes in one
env, or one settled pile swept many ways) and must never be split across
train/val.

Run: /home/alon/anaconda3/envs/pme/bin/python -u datasets/DS-0002-real-distinct-states/build.py
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from Genesis.binned_slate_dataset import BinnedSlateCorpus

ROOT = "/home/alon/Code/pile_manipulation"
DATA = os.path.join(ROOT, "Genesis/data")
OUT_DIR = os.path.join(ROOT, "datasets/DS-0002-real-distinct-states/data")
os.makedirs(OUT_DIR, exist_ok=True)

DEDUP_TOL_M = 0.001  # 1 mm -- matches sim position tolerance order of magnitude
SLATE_POST_SAMPLE = 5
RNG = np.random.default_rng(0)

rows = []  # each: dict with xy(np (n,2)) etc -- collected then batched


def add_rows(states_np, n_objects, corpus, source_file, spawn_mode, role, group_prefix,
             group_per_row=None):
    """states_np: (E, n_objects, 7) numpy. group_per_row: optional (E,) list of
    group ids; default is one group for the whole call (group_prefix)."""
    E = states_np.shape[0]
    for e in range(E):
        rows.append({
            "xyz": states_np[e, :, :3].astype(np.float32),
            "quat": states_np[e, :, 3:7].astype(np.float32),
            "n_objects": n_objects,
            "corpus": corpus,
            "source_file": source_file,
            "spawn_mode": spawn_mode,
            "role": role,
            "group": group_prefix if group_per_row is None else group_per_row[e],
        })


t0 = time.time()
skipped = []

# ---------------------------------------------------------------------------
# 1) overnight_randlen_{train,test} -- independent transitions
# ---------------------------------------------------------------------------
for split in ["train", "test"]:
    root = os.path.join(DATA, f"overnight_randlen_{split}")
    subdirs = sorted(glob.glob(f"{root}/*/"))
    for sd in subdirs:
        spawn_dir = os.path.basename(sd.rstrip("/"))  # e.g. mixed_n20, piled_n50
        files = sorted(glob.glob(f"{sd}*_data.pt"))
        for f in files:
            d = torch.load(f, map_location="cpu")
            states = d["states"].numpy()
            states_ = d["states_"].numpy()
            n_obj = states.shape[1]
            stem = os.path.relpath(f, ROOT)
            add_rows(states, n_obj, f"overnight_randlen_{split}", stem, spawn_dir,
                     "initial", f"randlen_{split}:{spawn_dir}:{os.path.basename(f)}")
            add_rows(states_, n_obj, f"overnight_randlen_{split}", stem, spawn_dir,
                     "post_sweep", f"randlen_{split}:{spawn_dir}:{os.path.basename(f)}")
print(f"[{time.time()-t0:.0f}s] after randlen: {len(rows)} rows")

# ---------------------------------------------------------------------------
# 2) Sean/<mode>-<ts>/<mode>/cube/... -- independent transitions, mixes n20/50/100
# ---------------------------------------------------------------------------
sean_files = sorted(glob.glob(f"{DATA}/Sean/*/*/cube/**/*_data.pt", recursive=True))
sean_files = [f for f in sean_files if "_failed" not in f]
for f in sean_files:
    # spawn mode is the path segment right after Sean/<ts-dirname>/
    parts = f.split(os.sep)
    mode = None
    for m in ("inbetween", "piled", "scattered"):
        if m in parts:
            mode = m
            break
    d = torch.load(f, map_location="cpu")
    states = d["states"].numpy()
    states_ = d["states_"].numpy()
    n_obj = states.shape[1]
    stem = os.path.relpath(f, ROOT)
    add_rows(states, n_obj, "Sean", stem, mode, "initial", f"sean:{mode}:{os.path.basename(f)}")
    add_rows(states_, n_obj, "Sean", stem, mode, "post_sweep", f"sean:{mode}:{os.path.basename(f)}")
print(f"[{time.time()-t0:.0f}s] after Sean ({len(sean_files)} files): {len(rows)} rows")

# ---------------------------------------------------------------------------
# 3) slates_binned/n20_scatter_s20a1000_L20-70mm -- SLATE-SHAPED
# ---------------------------------------------------------------------------
binned_dir = f"{DATA}/slates_binned/n20_scatter_s20a1000_L20-70mm"
c = BinnedSlateCorpus.load(binned_dir)
n_slates = int(c.slate_idx.max().item()) + 1 if hasattr(c, "slate_idx") else None
all_sel = c.select()  # everything
slate_idx_arr = all_sel.slate_idx.numpy()
n_slates = int(slate_idx_arr.max()) + 1
for s in range(n_slates):
    mask = slate_idx_arr == s
    idxs = np.nonzero(mask)[0]
    # initial state: same across the whole slate -- take row 0's `states`
    init_state = all_sel.states[idxs[0]].numpy()[None, ...]
    add_rows(init_state, init_state.shape[1], "slates_binned", binned_dir, "scatter",
             "initial", f"slates_binned:slate{s}")
    # post-sweep sample: <=5 states_ sampled from this slate's candidates
    take = RNG.choice(idxs, size=min(SLATE_POST_SAMPLE, len(idxs)), replace=False)
    post_states = all_sel.states_[take].numpy()
    add_rows(post_states, post_states.shape[1], "slates_binned", binned_dir, "scatter",
             "post_sweep", f"slates_binned:slate{s}")
print(f"[{time.time()-t0:.0f}s] after slates_binned ({n_slates} slates): {len(rows)} rows")

# ---------------------------------------------------------------------------
# 4) slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm} -- SLATE-SHAPED
#    (base dirs only; *_eval and *_train are filtered/aliased views, skipped)
# ---------------------------------------------------------------------------
for tag in ["n20_L10mm", "n20_L20mm", "n20_L40mm"]:
    root = f"{DATA}/slates_multistep/{tag}"
    manifest_path = f"{root}/manifest.json"
    if not os.path.exists(manifest_path):
        skipped.append((f"slates_multistep/{tag}", "no manifest.json found"))
        continue
    manifest = json.load(open(manifest_path))
    batches = manifest["batches"]
    # group batches by slate_idx; step 0 is the initial (shared, broadcast) state
    by_slate = {}
    for b in batches:
        by_slate.setdefault(b["slate_idx"], []).append(b)
    for slate_idx, blist in by_slate.items():
        step0 = [b for b in blist if b["step_idx"] == 0]
        if not step0:
            skipped.append((f"slates_multistep/{tag} slate {slate_idx}", "no step0 batch"))
            continue
        b0 = step0[0]
        f0 = f"{root}/_{b0['batch_idx']}_data.pt"
        if not os.path.exists(f0):
            continue
        d0 = torch.load(f0, map_location="cpu")
        init_state = d0["states"][0:1].numpy()  # env 0's initial state, same for the whole slate
        add_rows(init_state, init_state.shape[1], "slates_multistep", f"{tag}/manifest.json",
                  "heap", "initial", f"slates_multistep:{tag}:slate{slate_idx}")
        # post-sweep sample across ALL steps/envs of this slate, <=5 total
        pool_states = []
        for b in blist:
            fb = f"{root}/_{b['batch_idx']}_data.pt"
            if not os.path.exists(fb):
                continue
            db = torch.load(fb, map_location="cpu")
            pool_states.append(db["states_"].numpy())
        if pool_states:
            pool = np.concatenate(pool_states, axis=0)
            take_n = min(SLATE_POST_SAMPLE, pool.shape[0])
            take = RNG.choice(pool.shape[0], size=take_n, replace=False)
            add_rows(pool[take], pool.shape[1], "slates_multistep", f"{tag}/manifest.json",
                      "heap", "post_sweep", f"slates_multistep:{tag}:slate{slate_idx}")
print(f"[{time.time()-t0:.0f}s] after slates_multistep: {len(rows)} rows")

# ---------------------------------------------------------------------------
# Corpora found but SKIPPED, with reasons (report in DATASET.md verbatim)
# ---------------------------------------------------------------------------
skipped += [
    ("Genesis/data/overnight_randlen (un-split)", "superseded by overnight_randlen_train/test per docs/CODEMAP.md; using the split versions avoids double-counting the same rows"),
    ("Genesis/data/chickpeas, chickpeas_no_shuffle_position", "spheres on glass/wood, .pkl payload not the (n,7) cube-pose schema DS-0002 targets"),
    ("Genesis/data/corl", "mixed sphere/cube legacy corpus, provenance/config not verified within budget"),
    ("Genesis/data/corl_limited", "cubes but at swept sizes (0.005/0.00675/0.0085) -- mixing cube_size would break the fixed-REST_Z/CUBE_SIZE schema DS-0002 and DS-0003 share; out of scope for this budget"),
    ("Genesis/data/cube_spectrum", "deliberate particle-size/count sweep (granularity spectrum), not representative single-condition states"),
    ("Genesis/data/dinowm_test, dinowm_test_dino_wm", "smoke-test corpora for a different (DINO-WM) pipeline"),
    ("Genesis/data/foresight", "smoke-test / probe corpus (pile_smoke, probe_dense), not a maintained training corpus"),
    ("Genesis/data/granularity", "particle-count/size sweep, same rationale as cube_spectrum"),
    ("Genesis/data/mpc_runs", "MPC rollout trajectories (evaluation artefacts), not a states corpus"),
    ("Genesis/data/slates (n20_heap_5mm)", "superseded single-push-length slate corpus per docs/CODEMAP.md (4 of 6 length bins starved); slates_binned/slates_multistep are the maintained slate corpora"),
]

print(f"[{time.time()-t0:.0f}s] total rows before dedup: {len(rows)}")

# ---------------------------------------------------------------------------
# Deduplicate within each n_objects group
# ---------------------------------------------------------------------------
from collections import defaultdict
by_n = defaultdict(list)
for i, r in enumerate(rows):
    by_n[r["n_objects"]].append(i)

keep_idx = []
n_collapsed_total = 0
dedup_report = {}
for n_obj, idxs in by_n.items():
    keys = []
    for i in idxs:
        xy = rows[i]["xyz"][:, :2]
        q = np.round(xy / DEDUP_TOL_M).astype(np.int64)
        # canonical: sort particles (permutation-invariant), pack to bytes
        combo = q[:, 0] * 1_000_003 + q[:, 1]
        combo.sort()
        keys.append(combo.tobytes())
    keys_arr = np.array(keys, dtype=object)
    _, first_idx = np.unique(keys_arr, return_index=True)
    kept = [idxs[j] for j in sorted(first_idx.tolist())]
    n_collapsed = len(idxs) - len(kept)
    n_collapsed_total += n_collapsed
    dedup_report[int(n_obj)] = {"raw": len(idxs), "distinct": len(kept), "collapsed": n_collapsed}
    keep_idx.extend(kept)

keep_idx = sorted(keep_idx)
print(f"[{time.time()-t0:.0f}s] distinct rows after dedup: {len(keep_idx)} "
      f"(collapsed {n_collapsed_total} of {len(rows)})")
print("per-n_objects dedup report:", json.dumps(dedup_report, indent=2))

# ---------------------------------------------------------------------------
# Pack into a padded tensor payload + a metadata table
# ---------------------------------------------------------------------------
MAX_N = max(r["n_objects"] for r in rows)
N = len(keep_idx)
xyz = np.zeros((N, MAX_N, 3), dtype=np.float32)
quat = np.zeros((N, MAX_N, 4), dtype=np.float32)
n_objects = np.zeros((N,), dtype=np.int64)
meta = {"corpus": [], "source_file": [], "spawn_mode": [], "role": [], "group": []}
for row_i, i in enumerate(keep_idx):
    r = rows[i]
    n = r["n_objects"]
    xyz[row_i, :n] = r["xyz"]
    quat[row_i, :n] = r["quat"]
    n_objects[row_i] = n
    meta["corpus"].append(r["corpus"])
    meta["source_file"].append(r["source_file"])
    meta["spawn_mode"].append(r["spawn_mode"] or "unknown")
    meta["role"].append(r["role"])
    meta["group"].append(r["group"])

payload = {
    "xyz": torch.from_numpy(xyz),
    "quat": torch.from_numpy(quat),
    "n_objects": torch.from_numpy(n_objects),
    "corpus": meta["corpus"],
    "source_file": meta["source_file"],
    "spawn_mode": meta["spawn_mode"],
    "role": meta["role"],
    "group": meta["group"],
    "dedup_tol_m": DEDUP_TOL_M,
    "n_raw_rows": len(rows),
    "n_distinct_rows": N,
    "dedup_report_per_n_objects": dedup_report,
    "skipped_corpora": skipped,
}
out_path = os.path.join(OUT_DIR, "states.pt")
torch.save(payload, out_path)
print(f"[{time.time()-t0:.0f}s] saved {out_path}: {N} distinct real states, MAX_N={MAX_N}")

with open(os.path.join(OUT_DIR, "build_summary.json"), "w") as fh:
    json.dump({
        "n_raw_rows": len(rows),
        "n_distinct_rows": N,
        "dedup_tol_m": DEDUP_TOL_M,
        "dedup_report_per_n_objects": dedup_report,
        "skipped_corpora": skipped,
        "elapsed_s": time.time() - t0,
    }, fh, indent=2)
print("done.")
