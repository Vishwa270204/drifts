"""Compare 'update' against 'keep' on the newest labelled rows (held out from training)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import evaluate, row_err, selection_score
from .settings import KEEP, MIN_FIT, ROW_WISE, UPDATE
from .strategies import STRATEGIES, build


@dataclass
class Decision:
    error: Optional[str] = None
    metric: str = ""
    rec: str = KEEP
    scores: dict = field(default_factory=dict)      # strategy -> held-out score (lower is better)
    errs: dict = field(default_factory=dict)        # strategy -> per-row held-out error
    hold: int = 0                                   # held-out rows
    n_new: int = 0                                  # labelled rows since drift began
    notes: list = field(default_factory=list)       # short reliability caveats

    def change_pct(self, name: str) -> float:
        """Relative change of a strategy's score vs keeping the current model (negative = better)."""
        k, s = self.scores.get(KEEP, np.nan), self.scores.get(name, np.nan)
        return (s - k) / k * 100 if np.isfinite(k) and np.isfinite(s) and k > 1e-12 else np.nan


def compute_decision(sess) -> Decision:
    Xo, yo, Xn, yn = sess.get_data()
    n = len(yn)
    if n < 2 * MIN_FIT:
        return Decision(error=f"Only {n} new labelled rows, not enough to compare models (need {2 * MIN_FIT}).")
    hold = int(np.clip(round(n * sess.cfg.hold_frac), MIN_FIT, n - MIN_FIT))
    Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
    task, metric = sess.task, sess.metric
    scores, errs = {}, {}
    for sp in STRATEGIES:
        try:
            p = build(sess, sp.name, (Xo, yo, Xt, yt)).predict(Xh)
            if task == "regression" and not np.all(np.isfinite(np.asarray(p, float))):
                raise ValueError("non-finite predictions")
            scores[sp.name] = selection_score(metric, evaluate(task, yh, p))
            errs[sp.name] = row_err(task, yh, p)
        except Exception:
            continue                                 # a failed candidate is simply not offered
    if KEEP not in scores:
        return Decision(error="The current model could not be evaluated on the recent data.")

    order = {sp.name: i for i, sp in enumerate(STRATEGIES)}
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    rec = min(valid, key=lambda k: (round(valid[k], 12), k != KEEP, order[k])) if valid else KEEP   # ties -> keep

    notes = []
    if task == "classification":
        share = max(np.mean(yh == v) for v in np.unique(yh))
        if share > 0.95:
            notes.append(f"test rows are {share:.0%} one class")
    if rec != KEEP and metric in ROW_WISE:
        d = errs[KEEP] - errs[rec]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        if d.mean() <= se:
            notes.append("the gain is within noise")
    if hold < 100:
        notes.append(f"small test sample ({hold} rows)")
    return Decision(metric=metric, rec=rec, scores=scores, errs=errs, hold=hold, n_new=n, notes=notes)
