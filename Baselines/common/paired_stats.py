"""Baselines/common/paired_stats.py -- paired, state-resampled model comparison.

Every benchmark in this repo scores several models on the SAME set of states
(slates), so a model-vs-model comparison is paired: state difficulty cancels
in a per-state difference and is not noise for that comparison. The noise that
matters is the model x state interaction, and the unit of replication is the
STATE -- goals / value functions evaluated on one state are not independent
replicates (average them within state first).

Input everywhere: `X` of shape (M models, S states), a per-state score that is
HIGHER = BETTER (flip a cost before calling, e.g. with
`Baselines.common.goals.improvement`). NaN cells are dropped pairwise.

Established by EXP-0026, which also explains why the older "between-model sd
of means vs between-state sd of means" power check is not a power check: the
between-state sd is a population property that does not shrink with S.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats


def variance_components(X: np.ndarray) -> dict:
    """Two-way additive decomposition (model + state), complete rows only.
    `residual_sd` is the model x state interaction -- the paired noise."""
    X = X[:, ~np.isnan(X).any(axis=0)]
    M, S = X.shape
    g = X.mean()
    m, s = X.mean(1), X.mean(0)
    R = X - m[:, None] - s[None, :] + g
    df = (M - 1) * (S - 1)
    res_var = (R ** 2).sum() / df if df > 0 else float("nan")
    F = (S * ((m - g) ** 2).sum() / (M - 1)) / res_var if res_var > 0 else float("nan")
    return dict(n_models=M, n_states=S,
                model_sd_of_means=float(m.std(ddof=1)),
                state_sd_of_means=float(s.std(ddof=1)),
                residual_sd=float(math.sqrt(res_var)),
                anova_F_model=float(F),
                anova_p_model=float(stats.f.sf(F, M - 1, df)) if F == F else float("nan"))


def friedman(X: np.ndarray) -> dict:
    """Rank-based global test of 'all models equal', blocked by state."""
    X = X[:, ~np.isnan(X).any(axis=0)]
    if X.shape[0] < 3:
        return dict(stat=float("nan"), p=float("nan"), kendall_w=float("nan"))
    st, p = stats.friedmanchisquare(*X)
    M, S = X.shape
    return dict(stat=float(st), p=float(p), kendall_w=float(st / (S * (M - 1))))


def _signflip_p(d: np.ndarray, rng, n_mc: int = 20000) -> float:
    """Two-sided sign-flip permutation p for mean(d) = 0 (exact if S <= 16)."""
    S = len(d)
    obs = abs(d.mean())
    if S <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=S)))
    else:
        signs = rng.choice([-1, 1], size=(n_mc, S))
    null = np.abs((signs * d[None, :]).mean(1))
    return float((null >= obs - 1e-15).mean())


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def required_n(sd_d: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """States needed for a two-sided paired t-test to detect a mean paired
    difference `delta` given the sd of per-state differences `sd_d`
    (iterated t-quantile solution, starting from the normal approximation)."""
    if not (sd_d > 0 and delta > 0):
        return 0 if sd_d == 0 else -1
    zb = stats.norm.ppf(power)
    n = ((stats.norm.ppf(1 - alpha / 2) + zb) * sd_d / delta) ** 2
    for _ in range(50):
        df = max(n - 1, 1)
        n_new = ((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd_d / delta) ** 2
        if abs(n_new - n) < 1e-3:
            break
        n = n_new
    return int(math.ceil(max(n, 2)))


def paired_comparison(X: np.ndarray, names: list[str], n_boot: int = 10000,
                      seed: int = 0, ci: float = 0.95) -> list[dict]:
    """Every model pair (i, j): mean of X[i]-X[j] over states, sd of that
    difference, a percentile bootstrap CI resampling states, a sign-flip
    permutation p, Holm-adjusted over all pairs, and i's per-state win count."""
    rng = np.random.default_rng(seed)
    out = []
    for i, j in itertools.combinations(range(len(names)), 2):
        d = X[i] - X[j]
        d = d[~np.isnan(d)]
        S = len(d)
        idx = rng.integers(0, S, size=(n_boot, S))
        boot = d[idx].mean(1)
        lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
        out.append(dict(a=names[i], b=names[j], n_states=S,
                        mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)),
                        ci_lo=float(lo), ci_hi=float(hi),
                        p_signflip=_signflip_p(d, rng),
                        wins_a=int((d > 0).sum()), wins_b=int((d < 0).sum())))
    for r, pa in zip(out, holm([r["p_signflip"] for r in out])):
        r["p_holm"] = pa
        r["resolved_ci"] = not (r["ci_lo"] <= 0 <= r["ci_hi"])
    return out


def rank_stability(X: np.ndarray, names: list[str], n_boot: int = 10000,
                   seed: int = 0) -> dict:
    """Bootstrap states; per model: P(ranked first) and the 95% interval of
    its rank (1 = best) under state resampling."""
    X = X[:, ~np.isnan(X).any(axis=0)]
    rng = np.random.default_rng(seed)
    M, S = X.shape
    idx = rng.integers(0, S, size=(n_boot, S))
    means = X[:, idx].mean(2)                      # (M, n_boot)
    ranks = (-means).argsort(0).argsort(0) + 1     # 1 = best
    return {n: dict(p_first=float((ranks[k] == 1).mean()),
                    rank_lo=int(np.quantile(ranks[k], 0.025)),
                    rank_hi=int(np.quantile(ranks[k], 0.975)))
            for k, n in enumerate(names)}


def power_table(pairs: list[dict], deltas: list[float], n_pairs: int | None = None) -> dict:
    """Required states per delta, from the MEDIAN per-pair `sd_diff`, at
    alpha 0.05 and at Bonferroni alpha over `n_pairs` (default: all pairs)."""
    sd = float(np.median([p["sd_diff"] for p in pairs]))
    sd_max = float(np.max([p["sd_diff"] for p in pairs]))
    k = n_pairs or len(pairs)
    return dict(median_sd_diff=sd, max_sd_diff=sd_max, n_pairs=k,
                required_n={f"{d:g}": dict(alpha_05=required_n(sd, d),
                                          bonferroni=required_n(sd, d, alpha=0.05 / k))
                            for d in deltas})
