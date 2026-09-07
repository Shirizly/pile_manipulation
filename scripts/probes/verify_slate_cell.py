"""Verify one multi-step slate cell from Genesis/data/slates_multistep/.

Checks the four properties the collection is supposed to guarantee, and that a
later session should re-check before trusting a cell:

  * structure -- 3 batches per slate, one per step, constant env count, and no
    unresolved action-resampling failures;
  * the SAME-STATE property -- every env of a slate starts from an identical
    pile (checked at step 0 only; steps 1-2 diverge by construction, since each
    env has executed its own push by then);
  * action uniqueness within a batch -- no two candidates sharing a start
    position within 1 mm AND a heading within 5 degrees;
  * realized vs commanded push length -- with pile_aware=False there is no
    contact-triggered early stop, so this should be constant; a nonzero
    short-push fraction means that assumption broke.

Usage:
    PYTHONPATH=. python scripts/probes/verify_slate_cell.py \
        Genesis/data/slates_multistep/n20_L20mm 0.020
"""
import json, sys, glob, math
import numpy as np, torch

root = sys.argv[1]
L_cmd = float(sys.argv[2])
m = json.load(open(f"{root}/manifest.json"))
b = m["batches"]
steps, slates = {}, set()
for e in b:
    steps[e["step_idx"]] = steps.get(e["step_idx"], 0) + 1
    slates.add(e["slate_idx"])
unresolved = sum(int(e.get("unresolved_after_resampling") or 0) for e in b)
print(f"{root}")
print(f"  batches {len(b)}  slates {len(slates)}  per-step {dict(sorted(steps.items()))}  "
      f"envs {sorted({e['env_count'] for e in b})}  unresolved_resamples {unresolved}")

by_slate_step = {(e["slate_idx"], e["step_idx"]): e["batch_idx"] for e in b}
spread_max, dup_total, lens, nfail, ntrans = 0.0, 0, [], 0, 0
for (sl, st), bi in sorted(by_slate_step.items()):
    d = torch.load(f"{root}/_{bi}_data.pt", map_location="cpu", weights_only=False)
    ps, pe = d["p_starts"], d["p_stops"]
    ntrans += ps.shape[0]
    if st == 0:   # same-state property only holds at the sequence start
        pos = d["states"][..., 0:3]
        spread_max = max(spread_max, float((pos - pos[0:1]).abs().max()))
    v = (pe[:, :2] - ps[:, :2])
    lens.append(v.norm(dim=1).numpy())
    head = torch.atan2(v[:, 1], v[:, 0])
    n = ps.shape[0]
    dp = (ps[:, None, :2] - ps[None, :, :2]).norm(dim=2)
    dh = (head[:, None] - head[None, :]).abs()
    dh = torch.minimum(dh, 2 * math.pi - dh)
    close = (dp < 1e-3) & (dh < math.radians(5.0))
    dup_total += int((close.sum() - n) // 2)
    f = f"{root}/_{bi}_failed.pt"
    if glob.glob(f):
        ff = torch.load(f, map_location="cpu", weights_only=False)
        nfail += int(next((v.shape[0] for v in ff.values() if hasattr(v, "shape")), 0))
lens = np.concatenate(lens)
print(f"  transitions {ntrans}   failed rows {nfail}")
print(f"  same-state spread across envs (step 0): {spread_max:.3e} m  [need < 1e-9]")
print(f"  duplicate (start,heading) pairs within a batch: {dup_total}  [need 0]")
print(f"  realized length: min {lens.min():.6f}  max {lens.max():.6f}  mean {lens.mean():.6f} "
      f"(commanded {L_cmd})")
print(f"  short pushes (<99% of commanded): {100*float((lens < 0.99*L_cmd).mean()):.2f}%")
