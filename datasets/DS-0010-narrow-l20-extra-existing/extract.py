"""DS-0010: existing matched-physics rows for the narrow domain, as EXTRA TRAINING data only
(user 2026-09-25: 18-22 mm rows may be used for training, never testing).
Filter (EXP-0047 definitions): n = 20, single layer before AND after (every cube centre
z < 15 mm), push length 18-22 mm, perpendicular within 0.1 deg. Sources: the overnight_randlen
TRAIN split (overnight_randlen_train symlink dirs -- never the test split) and Sean n20.
Output: Genesis/data/narrow_l20_n20/extra_18_22/_{k}_data.pt + _{k}_config.yaml (the source
file's config), one output file per source file with >= 1 kept row. Resumable per file."""
import glob, math, os, shutil, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[2]
D = REPO / "Genesis/data"
OUT = D / "narrow_l20_n20/extra_18_22"; OUT.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(f"{D}/overnight_randlen_train/*n20/*_data.pt")) + \
        sorted(f for f in glob.glob(f"{D}/Sean/*/*/cube/n20/**/*_data.pt", recursive=True) if not f.endswith("_failed.pt"))
k, kept_total, seen = 0, 0, 0
man = []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    if "states" not in d or d["states"].shape[1] != 20:
        continue
    s, s_ = d["states"].float(), d["states_"].float()
    L = (d["p_stops"][:, :2] - d["p_starts"][:, :2]).float().norm(dim=1)
    dirn = torch.atan2(*(d["p_stops"][:, 1::-1] - d["p_starts"][:, 1::-1]).float().T.flip(0)) if False else \
        torch.atan2((d["p_stops"][:, 1] - d["p_starts"][:, 1]).float(), (d["p_stops"][:, 0] - d["p_starts"][:, 0]).float())
    err = torch.remainder(d["angles"].float() - dirn - math.pi / 2 + math.pi / 2, math.pi) - math.pi / 2
    keep = (L >= 0.018) & (L <= 0.022) & (err.abs() <= math.radians(0.1)) & \
           (s[:, :, 2].max(1).values < 0.015) & (s_[:, :, 2].max(1).values < 0.015)
    seen += len(L)
    if keep.sum() == 0:
        continue
    out = {x: d[x][keep] for x in ("states", "states_", "p_starts", "p_stops", "angles")}
    out["source_file"] = f; out["source_rows"] = keep.nonzero()[:, 0]
    dst = OUT / f"_{k}_data.pt"
    tmp = Path(str(dst) + ".tmp"); torch.save(out, tmp); os.replace(tmp, dst)
    cfg = f.replace("_data.pt", "_config.yaml")
    shutil.copyfile(os.path.realpath(cfg), OUT / f"_{k}_config.yaml")
    man.append(dict(k=k, source=f, rows=int(keep.sum()))); kept_total += int(keep.sum()); k += 1
print(f"scanned {seen} rows in {len(files)} files; kept {kept_total} rows in {k} files -> {OUT}")
import json; (OUT / "manifest.json").write_text(json.dumps(dict(kept=kept_total, files=man), indent=1))
