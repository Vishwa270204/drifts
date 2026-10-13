"""Error and scoring helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def ref_stats(task, e):
    """Mean and spread of per-row error on reference data; the drift threshold is built from these."""
    m, s = float(e.mean()), float(e.std())
    s = max(s, 0.1) if task == "classification" else s
    return m, max(s, 0.1 * m, 1e-9)


def auto_metric(task, y):
    """Error rate, unless one class is rare (<20%): then error rate hides misses, so use macro F1."""
    if task == "regression":
        return "MAE"
    _, cnt = np.unique(y, return_counts=True)
    return "1 - macro F1" if len(cnt) > 1 and cnt.min() / cnt.sum() < 0.2 else "Error rate"


def evaluate(task, y, p):
    y, p = np.asarray(y), np.asarray(p)
    if task == "regression":
        return dict(mae=float(np.abs(y.astype(float) - p.astype(float)).mean()))
    labels = np.unique(np.concatenate([y, p]))
    return dict(err=float((y != p).mean()),
                macro_f1=float(f1_score(y, p, labels=labels, average="macro", zero_division=0)))


def selection_score(metric, m):
    """Lower is better for every metric."""
    return {"Error rate": lambda: m["err"], "1 - macro F1": lambda: 1.0 - m["macro_f1"],
            "MAE": lambda: m["mae"]}[metric]()


def fmt_score(metric, v):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.1%}" if metric == "Error rate" else f"{v:.3f}"
