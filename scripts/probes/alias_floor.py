"""Observation-aliasing floor: how much of a push's effect is even DECIDABLE
from the model's input?

Every model in the repo is scored against persistence and mean-delta, but
nobody has measured the ceiling. In a deterministic simulator the outcome is a
function of the FULL state; a model sees a projected image plus an action. Two
transitions whose inputs are (nearly) identical but whose outcomes differ tell
us how much outcome variance the representation cannot resolve. That variance
upper-bounds every model that consumes that representation, and it is a
property of the REPRESENTATION x REGIME, not of the model class.

Estimator: for held-out pairs matched by input distance, E||d_i - d_j||^2 is
2*sigma^2_alias at zero input distance. Regress on input distance and read the
intercept (standard nearest-neighbour Bayes-error estimator).
"""
import argparse, torch
from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import canonicalise, actions_to_pixels

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--view", default="mask")
ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--min-grains", type=float, default=1.0)
ap.add_argument("--cube-size", type=float, default=None)
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--max-episodes", type=int, default=None)
ap.add_argument("--sub", type=int, default=4000, help="subsample transitions")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--min-push-mm", type=float, default=19.9)
a = ap.parse_args()

occ0, occ1, act, ep, s0, s1 = load_transition_fields(
    a.glob, a.grid, a.blur, "mean", a.min_push_mm, "cpu", view=a.view,
    min_grains=a.min_grains, cube_size=a.cube_size, max_episodes=a.max_episodes)

M = occ0.shape[0]
if M > a.sub:
    idx = torch.randperm(M, generator=torch.Generator().manual_seed(0))[:a.sub]
    occ0, occ1, act, ep = occ0[idx], occ1[idx], act[idx], ep[idx]
    M = a.sub

s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                               (BOUNDS["x_max"], BOUNDS["y_max"]), (a.grid, a.grid))
Y0 = canonicalise(occ0, s_px, e_px, a.res, a.crop).reshape(M, -1)
Y1 = canonicalise(occ1, s_px, e_px, a.res, a.crop).reshape(M, -1)
D = Y0.shape[1]
Δ = Y1 - Y0

# variance about the MEAN DELTA -- i.e. the quantity the zero-parameter
# baseline already explains is excluded, so ceiling R2 is directly comparable
# to the "explained over mean-delta" margins in docs/sand_manipulation.md.
var_delta = float(((Δ - Δ.mean(0, keepdim=True)) ** 2).sum(1).mean() / D)

# pair only ACROSS episodes: within an episode consecutive states are the same
# pile and would make the input distance small for a trivial reason.
dist = torch.cdist(Y0, Y0) / (D ** 0.5)
same = ep[:, None] == ep[None, :]
dist = dist.masked_fill(same, float("inf"))
dn, jn = dist.topk(a.k, largest=False, dim=1)

din2 = (dn ** 2).reshape(-1)
dout2 = (((Δ[:, None, :] - Δ[jn]) ** 2).sum(-1) / D).reshape(-1)

# robust linear fit of dout2 on din2, on the closest 60% of pairs (the far tail
# is where the local-linearity of the extrapolation breaks down)
q = torch.quantile(din2, 0.6)
m = din2 <= q
x, y = din2[m], dout2[m]
X = torch.stack([torch.ones_like(x), x], 1)
beta = torch.linalg.lstsq(X, y[:, None]).solution.squeeze(1)
c, slope = float(beta[0]), float(beta[1])
sigma2 = max(c, 0.0) / 2.0
ss = 1 - float(((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())

# what a k=1 retrieval model actually achieves, for reference
knn_mse = float(dout2.reshape(M, a.k)[:, 0].mean()) / 2.0 * 2.0  # ||d_i - d_j||^2

print(f"\n### {a.label or a.glob}  view={a.view} blur={a.blur} res={a.res}")
print(f"M={M} D={D} episodes={ep.unique().numel()}")
print(f"input NN distance (rms/px): p1={float(torch.quantile(dn[:,0],0.01)):.4f} "
      f"med={float(dn[:,0].median()):.4f} p99={float(torch.quantile(dn[:,0],0.99)):.4f}")
print(f"var(delta about mean-delta)      = {var_delta:.6f}")
print(f"aliasing floor sigma^2 (intercept)= {sigma2:.6f}   (fit R2={ss:.3f}, slope={slope:.3f})")
print(f"CEILING explained-over-mean-delta = {1 - sigma2/var_delta:.3f}")
print(f"k=1 retrieval outcome disagreement= {knn_mse:.6f}  -> naive R2 {1-knn_mse/2/var_delta:.3f}")
