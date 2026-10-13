"""Hoeffding Adaptive Tree wrapper with the small amount of preprocessing it needs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import MIN_FIT, Config

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:                                   # reported by the UI
    river_tree, river_drift, RIVER_OK = None, None, False


def make_tree(task, cat, cfg: Config):
    kw = dict(grace_period=int(cfg.grace), delta=float(cfg.delta), tau=float(cfg.tau),
              nominal_attributes=list(cat) or None, drift_window_threshold=int(cfg.win_thr),
              drift_detector=river_drift.ADWIN(delta=float(cfg.adwin)), seed=int(cfg.seed))
    if task == "regression":
        if cat:   # the default adaptive leaf uses a linear model, which cannot take string-valued features
            kw["leaf_prediction"] = "mean"
        return river_tree.HoeffdingAdaptiveTreeRegressor(**kw)
    return river_tree.HoeffdingAdaptiveTreeClassifier(**kw)


class Bundle:
    """Tree + median imputation for numeric columns; categorical columns are passed as nominal attributes."""

    def __init__(self, task, num, cat, cfg: Config):
        self.task, self.num, self.cat, self.cfg = task, list(num), list(cat), cfg
        self.cols = self.num + self.cat
        self.med, self.major, self.tree = {}, None, None

    def _rows(self, X):
        Z = X[self.cols]
        if self.num:
            Z = Z.copy()
            Z[self.num] = Z[self.num].fillna(self.med)
        return Z.to_dict("records")

    def _learn(self, rows, ys):
        reg = self.task == "regression"
        for x, t in zip(rows, ys):
            self.tree.learn_one(x, float(t) if reg else str(t))

    def fit(self, X, y):
        """Initial tree, trained in one pass in the given (time) order."""
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("features and target have different lengths")
        if len(y) < MIN_FIT:
            raise ValueError(f"only {len(y)} usable labelled rows (need at least {MIN_FIT})")
        if self.task == "regression":
            if not np.all(np.isfinite(y.astype(float))):
                raise ValueError("target contains non-finite values")
        else:
            self.major = str(pd.Series(y).mode().iloc[0])
        self.med = {c: (float(X[c].median()) if X[c].notna().any() else 0.0) for c in self.num}
        self.tree = make_tree(self.task, self.cat, self.cfg)
        self._learn(self._rows(X), y)
        return self

    def learn(self, X, y):
        """Keep learning on the existing tree, one row at a time."""
        y = np.asarray(y)
        if len(y) == 0:
            raise ValueError("no labelled rows to learn from")
        self._learn(self._rows(X), y)
        return self

    def predict(self, X):
        reg = self.task == "regression"
        out = []
        for x in self._rows(X):
            p = self.tree.predict_one(x)
            out.append((0.0 if p is None else float(p)) if reg else (self.major if p is None else str(p)))
        return np.asarray(out, float) if reg else np.asarray(out).astype(str)
