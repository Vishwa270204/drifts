"""State and behaviour of one monitoring run. No UI code.

    sess = Session.create(df, target="y", tcol=None, task="classification", cfg=Config.simple())
    sess.run(None)                       # until a decision is needed or data ends
    if sess.ready:
        sess.ensure_decision()
        ok, msg = sess.apply(sess.cand.rec)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data import build_frame, label_str, pick_features
from .decision import Decision, compute_decision
from .metrics import auto_metric, ref_stats, row_err
from .model import RIVER_OK, Bundle
from .settings import KEEP, METRICS, Config
from .strategies import build


@dataclass
class Alert:
    start: int                      # batch index where the suspected drift began
    status: str = "waiting"         # waiting | ready


@dataclass
class Retrain:
    version: int
    strategy: str
    applied: int                    # batch index when applied
    start: int                      # batch index where drift began


@dataclass
class Session:
    X: pd.DataFrame
    y: np.ndarray
    task: str
    cfg: Config
    n_tr: int
    N: int                          # rows in the streaming part
    nb: int                         # number of batches
    metric: str
    model: Bundle
    ref_m: float
    ref_s: float
    version: int = 0
    k: int = 0                      # batches received so far
    errs: list = field(default_factory=list)
    thr: list = field(default_factory=list)
    ver: list = field(default_factory=list)
    bad: list = field(default_factory=list)
    good: int = 0
    alert: Optional[Alert] = None
    snooze_until: int = 0
    cand: Optional[Decision] = None
    events: list = field(default_factory=list)
    retrains: list = field(default_factory=list)

    # ---- construction
    @classmethod
    def create(cls, df, target, tcol, task, cfg: Config) -> "Session":
        """Raises ValueError with a readable message on bad input."""
        if not RIVER_OK:
            raise ValueError("The `river` package is not installed (pip install river).")
        d = df.copy()
        if tcol not in (None, "(row order)"):
            tv = pd.to_numeric(d[tcol], errors="coerce") if pd.api.types.is_numeric_dtype(d[tcol]) \
                else pd.to_datetime(d[tcol], errors="coerce", utc=True)
            d = d.loc[tv.notna()].assign(_t=tv[tv.notna()]).sort_values("_t", kind="stable").drop(columns="_t")
        d = d[d[target].notna()]
        if task == "regression":
            yv = pd.to_numeric(d[target], errors="coerce")
            ok = np.isfinite(yv)
            d, y = d.loc[ok].reset_index(drop=True), yv[ok].to_numpy(float)
        else:
            d = d.reset_index(drop=True)
            y = label_str(d[target])
        B = int(cfg.B)
        n_tr = int(len(d) * cfg.train_pct / 100)
        n_fit = int(n_tr * 0.8)
        if len(d) < 400:
            raise ValueError("Need at least 400 labelled rows.")
        if n_fit < 50 or len(d) - n_tr < 2 * B:
            raise ValueError("Batch size is too large for this file. Lower it so the stream has at least 2 batches.")
        num, cat = pick_features(d.iloc[:n_tr], {target, tcol})   # chosen from the initial training rows only
        if not (num or cat):
            raise ValueError("No usable feature columns were found.")
        X = build_frame(d, num, cat)
        metric = cfg.metric if cfg.metric in METRICS[task] else auto_metric(task, y[:n_tr])
        try:
            model = Bundle(task, num, cat, cfg).fit(X.iloc[:n_fit], y[:n_fit])
        except Exception as exc:
            raise ValueError(f"Could not train the first model: {exc}")
        e = row_err(task, y[n_fit:n_tr], model.predict(X.iloc[n_fit:n_tr]))
        ref_m, ref_s = ref_stats(task, e)
        N = len(y) - n_tr
        return cls(X=X, y=y, task=task, cfg=cfg, n_tr=n_tr, N=N, nb=int(np.ceil(N / B)), metric=metric,
                   model=model, ref_m=ref_m, ref_s=ref_s)

    # ---- state queries
    @property
    def ready(self) -> bool:
        return self.alert is not None and self.alert.status == "ready"

    @property
    def done(self) -> bool:
        return self.k >= self.nb

    @property
    def can_advance(self) -> bool:
        return not (self.ready or self.done)

    @property
    def status(self) -> str:
        if self.ready:
            return "Decision needed"
        if self.done:
            return "Finished"
        return "Collecting data" if self.alert else "Monitoring"

    @property
    def new_rows(self) -> int:
        """Labelled rows available since the suspected drift start (respects label delay)."""
        al = self.alert
        if al is None:
            return 0
        known = self.k - self.cfg.delay
        return max(0, min(known * self.cfg.B, self.N) - al.start * self.cfg.B)

    @property
    def unit(self) -> str:
        return "MAE" if self.task == "regression" else "error rate"

    # ---- streaming engine
    def _log(self, event, detail):
        self.events.append({"Batch": self.k, "Event": event, "Detail": detail})

    def _clear_alert(self):
        self.alert, self.cand, self.bad, self.good = None, None, [], 0

    def step(self):
        """Receive one batch: predict with the deployed model, then judge the batch whose labels just arrived."""
        cfg, k = self.cfg, self.k
        a = self.n_tr + k * cfg.B
        b = min(a + cfg.B, len(self.y))
        if b <= a:
            self.k = self.nb
            return
        e = row_err(self.task, self.y[a:b], self.model.predict(self.X.iloc[a:b]))
        self.errs.append(float(e.mean()))
        self.thr.append(self.ref_m + cfg.z * self.ref_s / np.sqrt(b - a))
        self.ver.append(self.version)
        self.k = k + 1
        j = k - cfg.delay                                      # batch whose labels arrive now
        if j >= 0 and self.ver[j] == self.version:             # only judge batches scored by the deployed model
            flagged = self.errs[j] > self.thr[j]
            if self.alert is None:
                self.bad = self.bad + [j] if flagged else []
                if len(self.bad) >= 2 and self.k > self.snooze_until:     # 2 bad batches in a row = alert
                    self.alert, self.cand, self.good = Alert(start=self.bad[0]), None, 0
                    self.bad = []
                    self._log("Drift alert", f"errors above threshold since batch {self.alert.start + 1}")
            elif self.alert.status == "waiting":
                self.good = 0 if flagged else self.good + 1
                if self.good >= 3:                                         # it went away by itself
                    self.alert = None
                    self._log("Alert cleared", "error returned to normal")
        if self.alert and self.alert.status == "waiting" and self.new_rows >= cfg.min_rows:
            self.alert.status = "ready"

    def run(self, n: Optional[int] = None):
        """Advance up to n batches (None = until a decision is needed or the data ends)."""
        i = 0
        while self.can_advance and (n is None or i < n):
            self.step()
            i += 1

    # ---- data access
    def get_data(self):
        """(old X, old y, new X, new y): old = labelled rows before the drift start, new = since."""
        B = self.cfg.B
        s0 = self.n_tr + self.alert.start * B
        s1 = max(s0, self.n_tr + min((self.k - self.cfg.delay) * B, self.N))
        return self.X.iloc[:s0], self.y[:s0], self.X.iloc[s0:s1], self.y[s0:s1]

    # ---- decisions
    def ensure_decision(self) -> Decision:
        """Computed once per alert; failures are stored as Decision(error=...)."""
        if self.cand is None:
            try:
                self.cand = compute_decision(self)
            except Exception as exc:
                self.cand = Decision(error=f"Could not compare models: {exc}")
        return self.cand

    def keep(self, detail: Optional[str] = None) -> str:
        """Dismiss the alert without changing the model; suppress new alerts for `snooze` batches."""
        n = self.cfg.snooze
        self.snooze_until = self.k + n
        self._log("Kept current model", detail or f"no alerts for {n} batches")
        self._clear_alert()
        return f"Kept the current model (v{self.version}). No new alerts for {n} batches."

    def apply(self, name):
        """Returns (ok, message). The deployed model and version change only on success."""
        if name == KEEP:
            return True, self.keep()
        c, al = self.cand, self.alert
        try:
            if not c or name not in c.errs:
                raise ValueError("this option was not evaluated")
            data = self.get_data()
            model = build(self, name, data)                # final build on ALL new labelled rows
            model.predict(data[2].iloc[:1])                # smoke test before accepting
        except Exception as exc:
            self._log("Update failed", f"{exc}. Deployed model unchanged.")
            return False, f"Update failed: {exc}. The deployed model was not changed."
        self.model = model
        self.version += 1
        self.ref_m, self.ref_s = ref_stats(self.task, c.errs[name])   # threshold from honest held-out errors
        self.retrains.append(Retrain(version=self.version, strategy=name, applied=self.k, start=al.start))
        self._log("Model updated", f"v{self.version} trained on {c.n_new:,} new rows")
        self._clear_alert()
        return True, f"Model updated to v{self.version}."

    # ---- reporting
    def retrain_summary(self) -> pd.DataFrame:
        """Live before/after error for every applied update."""
        ver, errs = np.asarray(self.ver), np.asarray(self.errs, float)
        rows = []
        for r in self.retrains:
            idx = np.arange(r.start, min(r.applied, len(errs)))
            before = errs[idx][ver[idx] == r.version - 1] if len(idx) else np.array([])
            after = errs[ver == r.version]
            b, a = (before.mean() if len(before) else np.nan), (after.mean() if len(after) else np.nan)
            rows.append({"Model": f"v{r.version}", "Updated at batch": r.applied + 1,
                         "Batches since": int(len(after)), "Error before": b, "Error after": a,
                         "Change (%)": (a - b) / b * 100 if np.isfinite(b) and b > 1e-12 and np.isfinite(a)
                         else np.nan})
        return pd.DataFrame(rows)
