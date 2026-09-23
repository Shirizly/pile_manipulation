"""A reusable value readout h(embedding, goal_embedding) -> scalar.

Experiment-local (EXP-0017): this is a COMPOSITION of sklearn pieces plus the
baseline protocol mandated by `experiments/METRICS.md::value_readout_r2`, not a
new project capability.

Accepts an ARBITRARY embedding vector for the state and an arbitrary embedding
for the goal -- the 87-dim analytic occupancy descriptor of EXP-0015 is just one
choice; a learned LeJEPA latent can be dropped in unchanged as long as it is
handed in as X_state (S, Ds) / X_goal (G, Dg).

Contract
--------
    ro = ValueReadout(capacity="mlp", feature_mode="concat")
    m  = ro.fit(X_state, X_goal, y, split)   # y is (S, G); split names row sets
    ro.predict(X_state_new, X_goal_new)      # (S', G') value matrix
    ro.save(path); ValueReadout.load(path)

`fit` ALWAYS computes the three mandatory baselines (`mean_only`, `state_only`,
`goal_only`) at the SAME capacity as the model, and reports
`increment_over_goal_only`. A readout that cannot beat `goal_only` cannot rank
actions, because within one slate the goal is constant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURE_MODES = ("diff", "concat", "state_only", "goal_only", "mean_only")
CAPACITIES = ("linear", "poly2", "mlp")


@dataclass
class ReadoutConfig:
    capacity: str = "mlp"
    feature_mode: str = "concat"
    hidden: tuple = (128, 128)
    mlp_alpha: float = 1e-3
    mlp_max_iter: int = 300
    ridge_alphas: tuple = tuple(np.logspace(-2, 6, 20))
    poly_alphas: tuple = tuple(np.logspace(-1, 7, 12))
    max_train_rows: int = 60000
    max_eval_rows: int = 40000
    seed: int = 0
    notes: str = ""
    provenance: dict = field(default_factory=dict)


class GroupRidgeCV(BaseEstimator, RegressorMixin):
    """Ridge whose alpha is chosen by GROUPED k-fold CV.

    Added for EXP-0018. `RidgeCV`'s default is leave-one-out generalized CV, i.e.
    ROW-random -- and post-push states inside one slate (or one trajectory) are
    near-duplicates, so a row-random inner fold puts a near-copy of every held-out
    row in the training part and selects a lambda orders of magnitude too small.
    That is the `readout-cv-folds-slate-aware` bug (EXP-0015 -> EXP-0017).
    Pass `groups` (one label per ROW) to `fit` and folds never split a group.
    """

    def __init__(self, alphas=(1.0,), n_splits=3):
        self.alphas = alphas
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        alphas = np.asarray(self.alphas, dtype=float)
        if groups is None:
            est = RidgeCV(alphas=alphas)
            est.fit(X, y)
            self.alpha_ = float(est.alpha_)
            self.cv_ = "row-random (RidgeCV LOO-GCV) -- NO grouping"
        else:
            groups = np.asarray(groups)
            n_splits = int(min(self.n_splits, len(np.unique(groups))))
            gkf = GroupKFold(n_splits=max(n_splits, 2))
            folds = list(gkf.split(X, y, groups))
            scores = np.zeros(len(alphas))
            for j, a in enumerate(alphas):
                acc = []
                for tr, te in folds:
                    m = Ridge(alpha=a).fit(X[tr], y[tr])
                    p = m.predict(X[te])
                    ss = float(np.sum((y[te] - p) ** 2))
                    st = float(np.sum((y[te] - y[tr].mean()) ** 2))
                    acc.append(-np.inf if st < 1e-12 else 1.0 - ss / st)
                scores[j] = float(np.mean(acc))
            self.alpha_ = float(alphas[int(np.argmax(scores))])
            self.cv_ = f"GroupKFold(n_splits={max(n_splits,2)}) over {len(np.unique(groups))} groups"
        self.est_ = Ridge(alpha=self.alpha_).fit(X, y)
        return self

    def predict(self, X):
        return self.est_.predict(X)


def _features(mode, Xs, Xg, s_idx, g_idx):
    """Rows are (state, goal) pairs given by the index arrays."""
    a = Xs[s_idx]
    b = Xg[g_idx]
    if mode == "mean_only":
        return np.zeros((len(s_idx), 0))
    if mode == "state_only":
        return a
    if mode == "goal_only":
        return b
    if mode == "diff":
        if a.shape[1] != b.shape[1]:
            raise ValueError("feature_mode='diff' needs matching state/goal dims; use 'concat'")
        return a - b
    if mode == "concat":
        return np.concatenate([a, b], axis=1)
    raise ValueError(mode)


def _r2(y_true, y_pred, train_mean):
    """R^2 with SS_tot taken about the TRAINING mean (METRICS.md)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - train_mean) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _make_estimator(cfg: ReadoutConfig):
    if cfg.capacity == "linear":
        return Pipeline([("scale", StandardScaler()),
                         ("ridge", GroupRidgeCV(alphas=np.asarray(cfg.ridge_alphas)))])
    if cfg.capacity == "poly2":
        return Pipeline([("scale0", StandardScaler()),
                         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                         ("scale1", StandardScaler()),
                         ("ridge", GroupRidgeCV(alphas=np.asarray(cfg.poly_alphas)))])
    if cfg.capacity == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=tuple(cfg.hidden),
                                          activation="relu", alpha=cfg.mlp_alpha,
                                          max_iter=cfg.mlp_max_iter, early_stopping=True,
                                          n_iter_no_change=10, random_state=cfg.seed))
    raise ValueError(cfg.capacity)


def _pairs(state_idx, goal_idx):
    ss, gg = np.meshgrid(state_idx, goal_idx, indexing="ij")
    return ss.reshape(-1), gg.reshape(-1)


class ValueReadout:
    def __init__(self, **kw):
        self.cfg = ReadoutConfig(**kw)
        self.est = None
        self.metrics = None

    # ---- core -------------------------------------------------------------
    def fit(self, X_state, X_goal, y, split, with_baselines=True, verbose=False,
            state_groups=None, goal_groups=None, cv_group_by=None):
        """split: dict with train_states/test_states/train_goals/test_goals (index arrays).

        Returns a metrics dict: the model's held-out R^2 plus the three mandatory
        baselines at the same capacity, and the increment over `goal_only`.

        `cv_group_by` (EXP-0018): "state" or "goal". When set, the inner ridge-lambda
        CV uses GroupKFold over `state_groups[row_state]` / `goal_groups[row_goal]`
        instead of row-random folds, so near-duplicate states (or repeats of one goal)
        cannot straddle a fold boundary. See `readout-cv-folds-slate-aware`.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        tr_s, tr_g = _pairs(split["train_states"], split["train_goals"])
        te_s, te_g = _pairs(split["test_states"], split["test_goals"])
        ytr_all = y[tr_s, tr_g]
        yte_all = y[te_s, te_g]
        train_mean = float(ytr_all.mean())

        if len(ytr_all) > cfg.max_train_rows:
            k = rng.choice(len(ytr_all), cfg.max_train_rows, replace=False)
            tr_s, tr_g, ytr = tr_s[k], tr_g[k], ytr_all[k]
        else:
            ytr = ytr_all
        if len(yte_all) > cfg.max_eval_rows:
            k = rng.choice(len(yte_all), cfg.max_eval_rows, replace=False)
            te_s, te_g, yte = te_s[k], te_g[k], yte_all[k]
        else:
            yte = yte_all

        if cv_group_by == "state":
            row_groups = np.asarray(state_groups)[tr_s]
        elif cv_group_by == "goal":
            row_groups = np.asarray(goal_groups)[tr_g] if goal_groups is not None else tr_g
        elif cv_group_by is None:
            row_groups = None
        else:
            raise ValueError(cv_group_by)

        def run(mode):
            Xtr = _features(mode, X_state, X_goal, tr_s, tr_g)
            Xte = _features(mode, X_state, X_goal, te_s, te_g)
            if Xtr.shape[1] == 0:
                return _r2(yte, np.full(len(yte), train_mean), train_mean), None
            est = _make_estimator(cfg)
            if row_groups is not None and isinstance(est, Pipeline) and "ridge" in est.named_steps:
                est.fit(Xtr, ytr, ridge__groups=row_groups)
            else:
                est.fit(Xtr, ytr)
            return _r2(yte, est.predict(Xte), train_mean), est

        r2_model, self.est = run(cfg.feature_mode)
        alpha_sel, cv_desc = None, None
        if isinstance(self.est, Pipeline) and "ridge" in self.est.named_steps:
            alpha_sel = getattr(self.est.named_steps["ridge"], "alpha_", None)
            cv_desc = getattr(self.est.named_steps["ridge"], "cv_", None)
        out = {"capacity": cfg.capacity, "feature_mode": cfg.feature_mode,
               "cv_group_by": cv_group_by, "alpha_selected": alpha_sel, "inner_cv": cv_desc,
               "r2": r2_model, "n_train_rows": int(len(ytr)), "n_eval_rows": int(len(yte)),
               "train_mean": train_mean}
        if with_baselines:
            for b in ("mean_only", "state_only", "goal_only"):
                out[f"baseline_{b}"] = run(b)[0] if b != cfg.feature_mode else r2_model
            out["increment_over_goal_only"] = out["r2"] - out["baseline_goal_only"]
        self.metrics = out
        if verbose:
            print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in out.items()}), flush=True)
        return out

    def predict(self, X_state, X_goal):
        """Full (S', G') value matrix for arbitrary state/goal embedding batches."""
        if self.est is None:
            raise RuntimeError("not fitted")
        s = np.arange(len(X_state)); g = np.arange(len(X_goal))
        ss, gg = _pairs(s, g)
        X = _features(self.cfg.feature_mode, X_state, X_goal, ss, gg)
        return self.est.predict(X).reshape(len(s), len(g))

    def dv(self, z_now, z_pred, x_goal):
        """Action ranking signal: h(z_pred, goal) - h(z_now, goal), per candidate."""
        z_now = np.atleast_2d(z_now); z_pred = np.atleast_2d(z_pred)
        g = np.atleast_2d(x_goal)
        return (self.predict(z_pred, g) - self.predict(z_now, g)).reshape(-1)

    # ---- persistence ------------------------------------------------------
    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.cfg), "estimator": self.est,
                     "metrics": self.metrics, "format": "EXP-0017.ValueReadout.v1"}, path)
        return path

    @classmethod
    def load(cls, path):
        blob = joblib.load(path)
        ro = cls(**blob["config"])
        ro.est = blob["estimator"]
        ro.metrics = blob["metrics"]
        return ro
