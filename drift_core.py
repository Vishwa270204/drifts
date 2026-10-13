"""Core logic for the Model Drift Dashboard (no Streamlit imports).

A labelled dataset is streamed in batches through ONE Hoeffding Adaptive Tree (river) that learns continuously.

Two separate ADWIN detectors exist and must not be confused:
  * the tree's INTERNAL ADWIN (cfg.tree_adwin), inside the tree's nodes; it only runs while the tree is learning and
    lets the tree grow alternate subtrees and swap out branches that stopped working. It is configured in
    make_tree() and is never touched by the dashboard;
  * the DASHBOARD ADWIN (cfg.adwin, `Session.det`), which only WATCHES the model's prediction errors and raises an
    informational alert. It never changes the model, and the model does not need it in order to learn. ADWIN has no
    numeric error threshold: it cuts its window when two sub-windows differ significantly.

Streaming workflow (per arriving batch k, with label delay d):
  1. predict batch k with the current tree (no label of batch k is read);
  2. the labels of batch j = k - d arrive: errors of the STORED predictions for batch j are computed, the dashboard
     ADWIN is fed once, and then the tree learns batch j exactly once (so every record is predicted before it is
     learned, and every error belongs to the tree state that made the prediction);
  3. nothing waits for a user decision: an alert is only recorded and shown.
After the last batch has arrived, the remaining delayed labels are released one batch per step.

    sess = Session.create(df, target, tcol, task, Config.simple(100, "Medium"))
    sess.run(None)
"""
from __future__ import annotations

import inspect
import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ======================================================================
# settings
# ======================================================================
# Drift sensitivity -> delta of the dashboard ADWIN. delta is a CONFIDENCE parameter (smaller = needs stronger
# evidence = fewer alerts, later). It is not an error threshold.
SENS = {"Low (fewer alerts)": 0.0005, "Medium": 0.002, "High (more alerts)": 0.01}
MIN_FIT = 30                    # minimum usable labelled rows to train a tree
METRICS = {"classification": ["Error rate", "1 - macro F1"], "regression": ["MAE"]}


@dataclass
class Config:
    train_pct: int = 40             # share of rows used for the initial model
    B: int = 100                    # batch size (rows)
    adwin: float = 0.002            # delta of the DASHBOARD drift detector (see SENS)
    adwin_input: str = "rows"       # "rows": feed each batch's row errors | "batch": one mean error per batch
    delay: int = 0                  # label delay (batches)
    alert_hold: int = 10            # batches a recent alert keeps the status line in "error increase detected"
    metric: Optional[str] = None    # None = chosen automatically
    seed: int = 42
    grace: int = 200                # Hoeffding Adaptive Tree internals (unchanged)
    delta: float = 1e-7
    tau: float = 0.05
    tree_adwin: float = 0.002       # delta of the tree's INTERNAL ADWIN (separate from `adwin`)
    win_thr: int = 300

    @classmethod
    def simple(cls, batch_size: int = 100, sensitivity: str = "Medium") -> "Config":
        """The two user-facing choices; the rest keeps its defaults."""
        return cls(B=int(batch_size), adwin=SENS[sensitivity])


# ======================================================================
# model
# ======================================================================
try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:                                   # reported by the UI
    river_tree, river_drift, RIVER_OK = None, None, False

warnings.filterwarnings("ignore", category=UserWarning, module="river")


def make_tree(task, cat, cfg: Config):
    kw = dict(grace_period=int(cfg.grace), delta=float(cfg.delta), tau=float(cfg.tau),
              nominal_attributes=list(cat) or None, drift_window_threshold=int(cfg.win_thr),
              drift_detector=river_drift.ADWIN(delta=float(cfg.tree_adwin)), seed=int(cfg.seed))
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
        """Keep learning on the existing tree, one row at a time (this is what lets HAT's own ADWIN operate)."""
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

    def adaptation_counters(self) -> Optional[dict]:
        """Cumulative counters kept by the tree itself, or None if this River version does not expose them.
        Only these counters are treated as evidence that the tree changed its structure."""
        names = {"pruned": "n_pruned_alternate_trees", "switched": "n_switch_alternate_trees"}
        out = {}
        for key, attr in names.items():
            v = getattr(self.tree, attr, None)
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                out[key] = int(v)
        return out or None

# ======================================================================
# metrics
# ======================================================================
def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


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
    """Lower is better for every metric (used for the per-batch performance history)."""
    return {"Error rate": lambda: m["err"], "1 - macro F1": lambda: 1.0 - m["macro_f1"],
            "MAE": lambda: m["mae"]}[metric]()


def fmt_score(metric, v):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.1%}" if metric == "Error rate" else f"{v:.3f}"

# ======================================================================
# data
# ======================================================================
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if not data or not data.strip():
        raise ValueError("The file is empty.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(io.BytesIO(data), encoding=enc)
            if out.shape[1] <= 1:
                out = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return out
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not read the file as CSV: {exc}") from exc
    raise ValueError("Could not read the file as CSV.")


def label_str(s):
    """Stable string labels: integral numbers become '0','1' (not '0.0','1.0')."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


def guess_task(df, target):
    numeric = pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target])
    return "regression" if (numeric and df[target].nunique() > 10) else "classification"


def detect_time_col(df, target):
    """A column that clearly encodes time order, or None (then row order is used)."""
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            return c
        if not re.search(r"time|date|stamp", str(c), re.I):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.is_monotonic_increasing:
                return c
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s.head(200), errors="coerce", utc=True)
        if parsed.notna().mean() > 0.95:
            return c
    return None


def build_frame(df, num, cat):
    parts = {}
    for c in num:
        v = np.array(pd.to_numeric(df[c], errors="coerce"), dtype=float)
        v[~np.isfinite(v)] = np.nan
        parts[c] = v
    for c in cat:
        parts[c] = np.where(df[c].notna(), df[c].astype(str), "(missing)")
    return pd.DataFrame(parts, index=df.index)


def pick_features(df, exclude):
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        s, n = df[c], df[c].notna().sum()
        if n == 0 or s.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not (s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))):  # skip id-like
                num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)
    return num, cat


# ======================================================================
# dashboard drift detector (ADWIN on the model's prediction errors)
# ======================================================================
class DetectorError(RuntimeError):
    """The dashboard's ADWIN drift detector could not be created or updated."""


def make_monitor(cfg: Config):
    """Fresh dashboard ADWIN. Independent of the tree's internal ADWIN.

    "rows" mode uses River's defaults (it receives hundreds of values). "batch" mode receives only one value per
    batch, so ADWIN is asked to check after every value (clock=1) with small minimum windows. Arguments the installed
    River version does not support are dropped instead of guessed.
    """
    if not RIVER_OK:
        raise DetectorError("The `river` package is not installed.")
    if cfg.adwin_input not in ("rows", "batch"):
        raise DetectorError(f"adwin_input must be 'rows' or 'batch', not {cfg.adwin_input!r}")
    if not 0.0 < float(cfg.adwin) < 1.0:
        raise DetectorError(f"ADWIN delta must be between 0 and 1, not {cfg.adwin!r}")
    want = {"delta": float(cfg.adwin)}
    if cfg.adwin_input == "batch":
        want.update(clock=1, min_window_length=3, grace_period=5)
    accepted = inspect.signature(river_drift.ADWIN.__init__).parameters
    try:
        det = river_drift.ADWIN(**{k: v for k, v in want.items() if k in accepted})
    except Exception as exc:
        raise DetectorError(f"Could not create the ADWIN detector: {exc}") from exc
    for attr in ("estimation", "width"):
        if not hasattr(det, attr):
            raise DetectorError(f"This River version's ADWIN has no `{attr}` attribute; please upgrade river.")
    return det


def adwin_update(det, x: float) -> bool:
    """Feed one value; True if ADWIN signalled drift. Works with both River APIs: newer versions expose
    `drift_detected` after update(); older ones return (self, in_drift)."""
    out = det.update(x)
    if hasattr(det, "drift_detected"):
        return bool(det.drift_detected)
    if isinstance(out, tuple) and len(out) == 2:
        return bool(out[1])
    raise DetectorError("Cannot read the drift signal from this River version's ADWIN.")


# ======================================================================
# session
# ======================================================================
SCHEMA = 4   # bump when Session's fields change; the UI discards sessions created by older code


@dataclass
class Alert:
    """One informational error-increase detection of the dashboard ADWIN. Never requires an action."""
    id: int                         # 1, 2, 3 ... in detection order (lets the UI show each alert once)
    batch: int                      # 0-based batch whose labels made ADWIN signal
    start: int                      # estimated first batch of the change (from ADWIN's window cut)
    rows_seen: int                  # stream rows that had arrived when the labels of that batch were released
    reference: float                # error level ADWIN treated as normal just before the signal
    recent: float                   # mean error of the signalling batch
    n_signals: int = 1              # upward ADWIN signals inside that batch


@dataclass
class Session:
    X: pd.DataFrame
    y: np.ndarray                   # simulated history; labels of batch j are only read when they are released
    task: str
    cfg: Config
    n_tr: int
    N: int                          # rows in the streaming part
    nb: int                         # number of batches
    metric: str
    model: Bundle                   # the one HAT; it learns continuously and is never replaced
    det: Any                         # dashboard ADWIN; never reset by an alert
    err_scale: float                # errors are divided by this and clipped to [0, 1] before ADWIN (1.0 = 0/1 errors)
    k: int = 0                      # batches that have arrived (and been predicted)
    rel: int = 0                    # batches whose labels have been released (scored, monitored, learned)
    learned: int = 0                # streamed rows the tree has learned from (excludes the initial training)
    preds: list = field(default_factory=list)       # predictions of each arrived batch, stored until labels arrive
    seen: list = field(default_factory=list)        # rows learned by the tree at the moment it predicted each batch
    errs: list = field(default_factory=list)        # mean error per scored batch (index = batch)
    scores: list = field(default_factory=list)      # selected metric per scored batch (lower is better)
    batch_e: list = field(default_factory=list)     # per-row errors of each scored batch
    last_fed: int = -1              # last batch given to the dashboard ADWIN; guarantees no batch is fed twice
    last_learned: int = -1          # last batch given to the tree; guarantees no batch is learned twice
    learn_failed: int = 0           # batches the tree could not learn from (see events)
    signals: list = field(default_factory=list)     # ADWIN detections: {"batch", "dir": "up" | "down"}
    levels: dict = field(default_factory=dict)      # batch -> ADWIN window mean (error units) after that batch
    alerts: list = field(default_factory=list)      # Alert records, in detection order
    adaptations: list = field(default_factory=list)  # {"batch", "switched", "pruned"}: only if the tree's own counters rose
    events: list = field(default_factory=list)
    adapt_prev: Optional[dict] = None               # tree counters seen after the previous batch

    # ---- construction
    @classmethod
    def create(cls, df, target, tcol, task, cfg: Config) -> "Session":
        """Raises ValueError with a readable message on bad input."""
        if not RIVER_OK:
            raise ValueError("The `river` package is not installed (pip install river).")
        if df is None or len(df) == 0:
            raise ValueError("The file has no rows.")
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} is not in the file.")
        if tcol not in (None, "(row order)") and tcol not in df.columns:
            raise ValueError(f"Time column {tcol!r} is not in the file.")
        if tcol == target:
            raise ValueError("The time column and the target column must be different.")
        if int(df[target].notna().sum()) == 0:
            raise ValueError("The target column contains no labels. Supervised error monitoring and supervised "
                             "incremental learning need the true outcomes, so they cannot run until labels exist.")
        d = df.copy()
        if tcol not in (None, "(row order)"):
            tv = pd.to_numeric(d[tcol], errors="coerce") if pd.api.types.is_numeric_dtype(d[tcol]) \
                else pd.to_datetime(d[tcol], errors="coerce", utc=True)
            d = d.loc[tv.notna()].assign(_t=tv[tv.notna()]).sort_values("_t", kind="stable").drop(columns="_t")
        d = d[d[target].notna()]
        if task == "regression":
            yv = pd.to_numeric(d[target], errors="coerce")
            ok = np.isfinite(yv)
            if not ok.any():
                raise ValueError("The target column has no numeric values, so it cannot be used for regression.")
            d, y = d.loc[ok].reset_index(drop=True), yv[ok].to_numpy(float)
        else:
            d = d.reset_index(drop=True)
            y = label_str(d[target])
        B = int(cfg.B)
        n_tr = int(len(d) * cfg.train_pct / 100)
        n_fit = int(n_tr * 0.8)
        if len(d) < 400:
            raise ValueError(f"Need at least 400 labelled rows (found {len(d)}).")
        if n_fit < 50 or len(d) - n_tr < 2 * B:
            raise ValueError("Batch size is too large for this file. Lower it so the stream has at least 2 batches.")
        num, cat = pick_features(d.iloc[:n_tr], {target, tcol})   # chosen from the initial training rows only
        if not (num or cat):
            raise ValueError("No usable feature columns were found.")
        X = build_frame(d, num, cat)                              # the target is never part of X
        metric = cfg.metric if cfg.metric in METRICS[task] else auto_metric(task, y[:n_tr])
        try:
            model = Bundle(task, num, cat, cfg).fit(X.iloc[:n_fit], y[:n_fit])
        except Exception as exc:
            raise ValueError(f"Could not train the first model: {exc}")
        # ADWIN expects values in [0, 1]. Classification errors already are; regression errors are scaled by a
        # typical error size measured on the part of the initial data the model did not train on.
        e = row_err(task, y[n_fit:n_tr], model.predict(X.iloc[n_fit:n_tr]))
        scale = 1.0 if task == "classification" else max(5.0 * float(e.mean()), 3.0 * float(e.std()), 1e-9)
        try:
            det = make_monitor(cfg)
        except DetectorError as exc:
            raise ValueError(str(exc))
        N = len(y) - n_tr
        sess = cls(X=X, y=y, task=task, cfg=cfg, n_tr=n_tr, N=N, nb=int(np.ceil(N / B)), metric=metric,
                   model=model, det=det, err_scale=scale)
        sess.adapt_prev = model.adaptation_counters()
        return sess

    # ---- state queries
    @property
    def done(self) -> bool:
        """All batches have arrived and all their labels have been released and learned."""
        return self.k >= self.nb and self.rel >= self.nb

    @property
    def can_advance(self) -> bool:
        return not self.done

    @property
    def labels_pending(self) -> int:
        """Arrived batches whose labels are still delayed."""
        return self.k - self.rel

    @property
    def rows_seen(self) -> int:
        return min(self.k * self.cfg.B, self.N)

    @property
    def status(self) -> str:
        if self.done:
            return "Finished"
        last = self.alerts[-1] if self.alerts else None
        if last is not None and (self.rel - 1) - last.batch < self.cfg.alert_hold:
            return "Error increase detected — automatic learning continues"
        return "Monitoring — automatic learning on"

    @property
    def unit(self) -> str:
        return "MAE" if self.task == "regression" else "error rate"

    def new_alerts(self, shown: int) -> list:
        """Alerts the UI has not displayed yet (`shown` = how many it already displayed)."""
        return self.alerts[shown:]

    # ---- streaming engine
    def _log(self, event, detail, batch: Optional[int] = None):
        self.events.append({"Batch": self.k if batch is None else batch, "Event": event, "Detail": detail})

    def _fmt(self, v: float) -> str:
        return f"{v:.1%}" if self.task == "classification" else f"{v:.3g}"

    def step(self):
        """Advance the simulation by one step: receive the next batch (predict it with the current tree), then
        release the labels that are due (batch j = k - delay), score the stored predictions, update the dashboard
        ADWIN once, and let the tree learn that batch once. Once all batches have arrived, a step only releases
        the next delayed labels."""
        if self.done:
            return
        cfg = self.cfg
        arrived = False
        if self.k < self.nb:
            a = self.n_tr + self.k * cfg.B
            b = min(a + cfg.B, len(self.y))
            self.preds.append(self.model.predict(self.X.iloc[a:b]))   # before any label of this batch is used
            self.seen.append(self.learned)                           # which tree state made these predictions
            self.k += 1
            arrived = True
        j = self.rel
        if j < self.k and (not arrived or self.k - 1 - j >= cfg.delay):
            self._release(j)

    def _release(self, j: int):
        """Labels of batch j arrive: score the predictions made earlier, monitor them, then learn the batch."""
        cfg = self.cfg
        a = self.n_tr + j * cfg.B
        b = min(a + cfg.B, len(self.y))
        yj, p = self.y[a:b], self.preds[j]
        e = row_err(self.task, yj, p)                      # errors of the predictions that were actually made
        self.batch_e.append(e)
        self.errs.append(float(e.mean()))
        self.scores.append(float(selection_score(self.metric, evaluate(self.task, yj, p))))
        self._monitor(j)
        if j > self.last_learned:                          # exactly once, and only after the batch was scored
            try:
                self.model.learn(self.X.iloc[a:b], yj)
                self.learned += len(yj)
                self._check_adaptation(j)
            except Exception as exc:
                self.learn_failed += 1
                self._log("Learning skipped", f"batch {j + 1} could not be learned: {exc}", j + 1)
            self.last_learned = j
        self.rel = j + 1

    def _check_adaptation(self, j: int):
        """Report a structural change only when the tree's own counters say it happened."""
        now = self.model.adaptation_counters()
        prev = self.adapt_prev
        self.adapt_prev = now
        if not now or not prev:
            return
        sw, pr = now.get("switched", 0) - prev.get("switched", 0), now.get("pruned", 0) - prev.get("pruned", 0)
        if sw > 0 or pr > 0:
            self.adaptations.append({"batch": j, "switched": max(sw, 0), "pruned": max(pr, 0)})
            self._log("Tree adapted", f"HAT's own counters report {max(sw, 0)} alternate-subtree switch(es) and "
                                      f"{max(pr, 0)} pruned alternate subtree(s) while learning batch {j + 1}", j + 1)

    def _monitor(self, j: int):
        """Feed batch j's errors to the dashboard ADWIN exactly once and record its signals.

        "rows" mode feeds the batch's row errors in order; "batch" mode feeds one mean error per batch. ADWIN is
        direction-blind (it also fires when error FALLS), so each signal is classed up/down by comparing the window
        mean before and after the update. Only "up" creates an alert. The detector and the model are left alone.
        """
        cfg = self.cfg
        if j <= self.last_fed:
            return
        e = self.batch_e[j]
        raw = np.asarray(e if cfg.adwin_input == "rows" else [np.mean(e)], float)
        fin = np.isfinite(raw)
        if not fin.all():
            self._log("Skipped values", f"{int((~fin).sum())} non-finite error value(s) in batch {j + 1} "
                                        "were not given to ADWIN", j + 1)
        ups, downs = [], []
        try:
            for i in np.flatnonzero(fin):
                before = float(self.det.estimation)
                if adwin_update(self.det, float(np.clip(raw[i] / self.err_scale, 0.0, 1.0))):
                    after = float(self.det.estimation)
                    (ups if after > before else downs).append((int(i), int(self.det.width), before * self.err_scale))
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"The ADWIN detector failed on batch {j + 1}: {exc}") from exc
        self.last_fed = j
        if float(self.det.width) > 0:                      # window mean = the error level ADWIN treats as normal
            self.levels[j] = float(self.det.estimation) * self.err_scale
        if ups:
            self.signals.append({"batch": j, "dir": "up"})
        if downs:
            self.signals.append({"batch": j, "dir": "down"})
        if ups:
            i, width, ref = ups[0]
            start = self._change_point(j, i, width)
            al = Alert(id=len(self.alerts) + 1, batch=j, start=start,
                       rows_seen=min((j + 1 + cfg.delay) * cfg.B, self.N), reference=ref,
                       recent=self.errs[j], n_signals=len(ups))
            self.alerts.append(al)
            self._log("Error increase", f"ADWIN detected a statistically significant increase in the {self.unit} "
                                        f"at batch {j + 1} ({self._fmt(ref)} before, {self._fmt(al.recent)} in this "
                                        f"batch); estimated start: batch {start + 1}. This may or may not be concept "
                                        "drift. The tree keeps learning on its own.", j + 1)

    def _change_point(self, j: int, i: int, width: int) -> int:
        """Batch where the change began. ADWIN's window cut is coarse (it keeps a few extra older values), so its
        estimate is a lower bound: the change is somewhere between it and the signal batch j. Within that range,
        pick the first batch whose error is above the midpoint between the error level just before the range and
        the error at the signal."""
        if self.cfg.adwin_input == "rows":
            est = (i - width + 1 + j * self.cfg.B) // self.cfg.B
        else:
            est = j - width + 1
        est = int(max(0, min(est, j)))
        before = self.errs[max(0, est - 5):est]
        if before:
            mid = (float(np.mean(before)) + self.errs[j]) / 2
            for b in range(est, j + 1):
                if self.errs[b] > mid:
                    return b
        return est

    def run(self, n: Optional[int] = None):
        """Advance up to n steps (None = until the stream is finished). Alerts never stop it."""
        i = 0
        while self.can_advance and (n is None or i < n):
            self.step()
            i += 1


Session.SCHEMA = SCHEMA
