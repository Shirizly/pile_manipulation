"""DS-0007: same-state pools extracted from Genesis/data/Sean (not used to train
any current model; physics = the training corpora's: particle friction 0.7,
box friction 0.5, density 450).

A pool = rows of one `_k_data.pt` file whose START state is identical (positions
rounded to 0.1 mm). Pools with >= MIN_POOL rows are kept. Output: one shard per
(spawn mode, n particles): data/<spawn>_n<N>.pt with flat rows
{states, states_, p_starts, p_stops, angles, pool_idx, file_idx} + files list.
Checkpoint: shards + manifest.json rewritten every 50 files (atomic).
"""
import glob, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "datasets/DS-0007-sean-same-state-pools/data"
MIN_POOL = 6

def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    if str(path).endswith(".json"): tmp.write_text(json.dumps(obj, indent=1))
    else: torch.save(obj, tmp)
    os.replace(tmp, path)

files = sorted(glob.glob(str(REPO / "Genesis/data/Sean/**/*_data.pt"), recursive=True))
shards = defaultdict(lambda: defaultdict(list))
meta = defaultdict(lambda: {"files": [], "n_pools": 0, "n_rows": 0, "pool_sizes": defaultdict(int)})
t0 = time.time()
for fi, f in enumerate(files):
    rel = os.path.relpath(f, REPO)
    spawn = re.search(r"Sean/[^/]+/([a-z]+)/cube", rel).group(1)
    x = torch.load(f, map_location="cpu", weights_only=False)
    s = x["states"].float(); n = s.shape[1]
    key = torch.round(s[..., :3] * 1e4).reshape(len(s), -1)
    u, inv, cnt = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    name = f"{spawn}_n{n}"
    m = meta[name]
    if rel not in m["files"]: m["files"].append(rel)
    fidx = m["files"].index(rel)
    for g in (cnt >= MIN_POOL).nonzero().flatten().tolist():
        rows = (inv == g).nonzero().flatten()
        pid = m["n_pools"]; m["n_pools"] += 1; m["n_rows"] += len(rows); m["pool_sizes"][int(len(rows))] += 1
        sh = shards[name]
        for k in ("states", "states_", "p_starts", "p_stops", "angles"):
            sh[k].append(x[k][rows].float())
        sh["pool_idx"].append(torch.full((len(rows),), pid, dtype=torch.long))
        sh["file_idx"].append(torch.full((len(rows),), fidx, dtype=torch.long))
    if (fi + 1) % 50 == 0 or fi + 1 == len(files):
        for nm, sh in shards.items():
            save_atomic({k: torch.cat(v) for k, v in sh.items()} | {"files": meta[nm]["files"]}, OUT / f"{nm}.pt")
        save_atomic({"files_done": fi + 1, "files_total": len(files), "min_pool": MIN_POOL,
                     "shards": {nm: {"n_pools": v["n_pools"], "n_rows": v["n_rows"], "n_files": len(v["files"]),
                                     "pool_sizes": dict(v["pool_sizes"])} for nm, v in meta.items()},
                     "complete": fi + 1 == len(files)}, OUT / "manifest.json")
        print(f"{fi + 1}/{len(files)} files, {time.time() - t0:.0f}s: " +
              ", ".join(f"{nm} {v['n_pools']} pools" for nm, v in sorted(meta.items())), flush=True)
