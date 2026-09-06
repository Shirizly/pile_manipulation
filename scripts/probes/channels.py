"""Does a HEIGHT/DENSITY channel add information a silhouette does not carry?

docs/sand_manipulation.md §10 Q1 is explicit that this is untested. It is
answerable offline: every dataset stores full particle positions, so any
projection can be recomputed without touching the simulator.

Design: for EACH target in {mask-delta, density-delta}, all six inputs are
scored on that same target so only the INPUT changes within a target column.
Both targets are fit from a single load pass (mask/height/density views are
all loaded regardless of which becomes the target), so running both targets
costs no extra data-loading time over running one.

EXP-0023/C-017 repeat: the original script (aac084e3) only ever fit the
mask-delta target, which the 2026-09-05 amendment to EXP-0005 flagged as a
live "target-matching" confound -- it cannot tell "depth carries no extra
information" from "whichever input matches the target wins". Added the
--target loop to settle it: if mask still wins under a density target, that is
depth-is-uninformative; if each view wins on its own target, that is
target-matching. Episode-level split. Metric is explained variance over the
zero-parameter mean-delta baseline, which is the baseline
docs/sand_manipulation.md §7 established as the one that matters.
"""
import argparse, torch
from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import canonicalise, actions_to_pixels

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--cube-size", type=float, default=None)
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--min-grains", type=float, default=1.0)
ap.add_argument("--max-episodes", type=int, default=None)
ap.add_argument("--min-push-mm", type=float, default=19.9)
ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

def load(view):
    return load_transition_fields(a.glob, a.grid, a.blur, "mean", a.min_push_mm, "cpu",
                            view=view, min_grains=a.min_grains,
                            cube_size=a.cube_size, max_episodes=a.max_episodes)

m0, m1, act, ep, _, _ = load("mask")
h0, h1, _, _, _, _ = load("height")
d0, d1, _, _, _, _ = load("density")
M = m0.shape[0]
s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                               (BOUNDS["x_max"], BOUNDS["y_max"]), (a.grid, a.grid))
can = lambda o: canonicalise(o, s_px, e_px, a.res, a.crop).reshape(M, -1)
Mk0, Mk1, Ht0, Dn0 = can(m0), can(m1), can(h0), can(d0)
# height maps are metres; rescale so the two channels have comparable energy and
# the shared ridge does not silently switch one of them off.
Ht0 = Ht0 / Ht0.std().clamp_min(1e-9) * Mk0.std()
Dn0 = Dn0 / Dn0.std().clamp_min(1e-9) * Mk0.std()

eps = ep.unique()
g = torch.Generator().manual_seed(a.seed)
val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps)//4)].tolist())
te = torch.tensor([int(e) in val for e in ep]); tr = ~te

Dk1 = can(d1)  # density-view target counterpart to Mk1

print(f"\n### {a.label or a.glob}  res={a.res} blur={a.blur} "
      f"M={M} ({int(tr.sum())} train / {int(te.sum())} val, {len(eps)} episodes)")

for target_name, T0, T1 in [("mask-delta", Mk0, Mk1), ("density-delta", Dn0, Dk1)]:
    target_tr, target_te = T1[tr] - T0[tr], T1[te] - T0[te]
    mu = target_tr.mean(0, keepdim=True)
    den = float(((target_te - mu) ** 2).mean())

    def run(name, Xtr, Xte):
        # predict the DELTA from the input, ridge toward zero delta (= mean-delta
        # once the bias column is included).
        ones = torch.ones(Xtr.shape[0], 1)
        A = torch.cat([Xtr, ones], 1)
        G = A.T @ A + a.ridge * torch.eye(A.shape[1])
        G[-1, -1] = A.shape[0] * 1e-9
        W = torch.linalg.solve(G, A.T @ target_tr)
        P = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1) @ W
        num = float(((target_te - P) ** 2).mean())
        print(f"  {name:28s} dim={Xtr.shape[1]:5d}  explained over mean-delta = {1-num/den:.4f}")

    print(f"\n--- target = {target_name} ---")
    print(f"  {'mean-delta (0 params)':28s} dim=    0  explained over mean-delta = 0.0000  (by construction)")
    run("mask only", Mk0[tr], Mk0[te])
    run("height only", Ht0[tr], Ht0[te])
    run("density only", Dn0[tr], Dn0[te])
    run("mask + height (2 channel)", torch.cat([Mk0[tr], Ht0[tr]], 1),
        torch.cat([Mk0[te], Ht0[te]], 1))
    run("mask + density (2 channel)", torch.cat([Mk0[tr], Dn0[tr]], 1),
        torch.cat([Mk0[te], Dn0[te]], 1))
    run("mask+height+density (3ch)", torch.cat([Mk0[tr], Ht0[tr], Dn0[tr]], 1),
        torch.cat([Mk0[te], Ht0[te], Dn0[te]], 1))
