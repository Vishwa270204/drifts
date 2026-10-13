"""Core logic for the Model Drift Dashboard (no Streamlit imports).

A labelled dataset is streamed in batches through one deployed Hoeffding Adaptive Tree (river).

Two separate ADWIN detectors exist and must not be confused:
  * the tree's INTERNAL ADWIN (cfg.tree_adwin), inside every node; it only runs while the tree is learning and
    swaps out branches that stopped working;
  * the DASHBOARD ADWIN (cfg.adwin, `Session.det`), which watches the deployed model's prediction errors and raises
    the drift alert. ADWIN has no numeric threshold: it cuts its window when two sub-windows differ significantly.

On an alert the user compares "update" against "keep" on held-out recent rows and decides.
    sess = Session.create(df, target, tcol, task, Config.simple(100, "Medium"))
    sess.run(None); sess.ensure_decision(); sess.apply(sess.cand.rec)
"""
from __future__ import annotations

import copy
import inspect
import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
UPDATE, KEEP = "Update model", "Keep current model"
METRICS = {"classification": ["Error rate", "1 - macro F1"], "regression": ["MAE"]}
ROW_WISE = {"Error rate", "MAE"}   # metrics with a per-row loss (paired noise check possible)


@dataclass
class Config:
    train_pct: int = 40             # share of rows used for the initial model
    B: int = 100                    # batch size (rows)
    adwin: float = 0.002            # delta of the DASHBOARD drift detector (see SENS)
    adwin_input: str = "rows"       # "rows": feed each batch's row errors | "batch": one mean error per batch
    delay: int = 0                  # label delay (batches)
    min_rows: int = 400             # new labelled rows needed before asking
    snooze: int = 10                # batches without alerts after "keep"
    hold_frac: float = 0.25         # share of new labelled rows held out for evaluation
    metric: Optional[str] = None    # None = chosen automatically
    seed: int = 42
    grace: int = 200                # Hoeffding Adaptive Tree internals (unchanged)
    delta: float = 1e-7
    tau: float = 0.05
    tree_adwin: float = 0.002       # delta of the tree's INTERNAL ADWIN (separate from `adwin`)
    win_thr: int = 300

    @classmethod
    def simple(cls, batch_size: int = 100, sensitivity: str = "Medium") -> "Config":
        """The two user-facing choices; the rest scales from them."""
        return cls(B=int(batch_size), adwin=SENS[sensitivity], min_rows=max(400, 4 * int(batch_size)))


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
    """Lower is better for every metric."""
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
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(io.BytesIO(data), encoding=enc)
            if out.shape[1] <= 1:
                out = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return out
        except UnicodeDecodeError:
            continue
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
# strategies
# ======================================================================
@dataclass
class Plan:
    kind: str                       # keep | continue
    X: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None


@dataclass
class Strategy:
    name: str
    plan: Callable


def plan_update(sess, Xo, yo, Xn, yn):
    return Plan("continue", Xn, yn)


def plan_keep(sess, Xo, yo, Xn, yn):
    return Plan("keep")


STRATEGIES = [Strategy(UPDATE, plan_update), Strategy(KEEP, plan_keep)]
REG = {s.name: s for s in STRATEGIES}


def build(sess, name, data):
    """Return a model for the strategy. Never mutates the deployed model (updates use a deep copy)."""
    plan = REG[name].plan(sess, *data)
    if plan.kind == "keep":
        return sess.model
    if plan.kind == "continue":
        return copy.deepcopy(sess.model).learn(plan.X, plan.y)
    raise ValueError(f"unknown plan kind {plan.kind}")

# ======================================================================
# decision
# ======================================================================
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


# ======================================================================
# dashboard drift detector (ADWIN on the deployed model's errors)
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
@dataclass
class Alert:
    start: int                      # estimated first batch of the change (from ADWIN's window cut)
    detected: int                   # batch where ADWIN signalled
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
    det: Any                        # dashboard ADWIN; reset on every model change or "keep"
    err_scale: float                # errors are divided by this and clipped to [0, 1] before ADWIN (1.0 = 0/1 errors)
    version: int = 0
    version_start: int = 0          # first batch scored by the deployed model version
    k: int = 0                      # batches received so far
    errs: list = field(default_factory=list)        # mean error per batch (for the chart)
    ver: list = field(default_factory=list)         # model version that scored each batch
    batch_e: list = field(default_factory=list)     # per-row errors of each batch (kept until labels arrive)
    last_fed: int = -1              # last batch given to ADWIN; guarantees no batch is fed twice
    signals: list = field(default_factory=list)     # ADWIN detections: {"batch", "dir": "up" | "down"}
    levels: dict = field(default_factory=dict)      # batch -> ADWIN window mean (error units) after that batch
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
        # ADWIN expects values in [0, 1]. Classification errors already are; regression errors are scaled by a
        # typical error size measured on the part of the initial data the model did not train on.
        e = row_err(task, y[n_fit:n_tr], model.predict(X.iloc[n_fit:n_tr]))
        scale = 1.0 if task == "classification" else max(5.0 * float(e.mean()), 3.0 * float(e.std()), 1e-9)
        try:
            det = make_monitor(cfg)
        except DetectorError as exc:
            raise ValueError(str(exc))
        N = len(y) - n_tr
        return cls(X=X, y=y, task=task, cfg=cfg, n_tr=n_tr, N=N, nb=int(np.ceil(N / B)), metric=metric,
                   model=model, det=det, err_scale=scale)

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
        """Labelled rows available since the estimated change (respects label delay)."""
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
        self.alert, self.cand = None, None

    def _reset_detector(self):
        """Start a new monitoring period: a fresh ADWIN, so no state from an old model or old alert leaks in."""
        self.det = make_monitor(self.cfg)

    def step(self):
        """Receive one batch: predict with the deployed model. Then, for the batch whose labels arrive now
        (j = k - delay) and only if the deployed model scored it, update the drift detector once."""
        cfg, k = self.cfg, self.k
        a = self.n_tr + k * cfg.B
        b = min(a + cfg.B, len(self.y))
        if b <= a:
            self.k = self.nb
            return
        e = row_err(self.task, self.y[a:b], self.model.predict(self.X.iloc[a:b]))
        self.batch_e.append(e)
        self.errs.append(float(e.mean()))
        self.ver.append(self.version)
        self.k = k + 1
        j = k - cfg.delay
        if j >= 0 and j > self.last_fed and self.ver[j] == self.version:
            self._monitor(j)
        if self.alert and self.alert.status == "waiting" and self.new_rows >= cfg.min_rows:
            self.alert.status = "ready"

    def _monitor(self, j: int):
        """Feed batch j's errors to ADWIN exactly once and act on its signals.

        "rows" mode feeds the batch's row errors in order; "batch" mode feeds one mean error per batch. ADWIN is
        direction-blind (it also fires when error FALLS), so each signal is classed up/down by comparing the window
        mean before and after the update. Only "up" raises an alert; "down" clears a waiting alert.
        """
        cfg = self.cfg
        e = self.batch_e[j]
        raw = np.asarray(e if cfg.adwin_input == "rows" else [np.mean(e)], float)
        fin = np.isfinite(raw)
        if not fin.all():
            self._log("Skipped values", f"{int((~fin).sum())} non-finite error value(s) in batch {j + 1} "
                                        "were not given to ADWIN")
        ups, downs = [], []
        try:
            for i in np.flatnonzero(fin):
                before = float(self.det.estimation)
                if adwin_update(self.det, float(np.clip(raw[i] / self.err_scale, 0.0, 1.0))):
                    (ups if float(self.det.estimation) > before else downs).append((int(i), int(self.det.width)))
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

        if self.alert is None:
            if ups and self.k > self.snooze_until:
                start = self._change_point(j, *ups[0])
                self.alert, self.cand = Alert(start=start, detected=j), None
                self._log("Drift alert", f"ADWIN flagged a rise in error at batch {j + 1}; "
                                         f"change estimated from batch {start + 1}")
        elif self.alert.status == "waiting" and downs and not ups:
            self._clear_alert()
            self._log("Alert cleared", "ADWIN saw the error fall back before retraining was needed")

    def _change_point(self, j: int, i: int, width: int) -> int:
        """Batch where the change began. ADWIN's window cut is coarse (it keeps a few extra older values), so its
        estimate is a lower bound: the change is somewhere between it and the signal batch j. Within that range,
        pick the first batch whose error is above the midpoint between the error level just before the range and
        the error at the signal. Never earlier than the batch where the deployed model began scoring."""
        if self.cfg.adwin_input == "rows":
            est = (i - width + 1 + j * self.cfg.B) // self.cfg.B
        else:
            est = j - width + 1
        est = int(max(self.version_start, min(est, j)))
        before = self.errs[max(self.version_start, est - 5):est]
        if before:
            mid = (float(np.mean(before)) + self.errs[j]) / 2
            for b in range(est, j + 1):
                if self.errs[b] > mid:
                    return b
        return est

    def run(self, n: Optional[int] = None):
        """Advance up to n batches (None = until a decision is needed or the data ends)."""
        i = 0
        while self.can_advance and (n is None or i < n):
            self.step()
            i += 1

    # ---- data access
    def get_data(self):
        """(old X, old y, new X, new y): old = labelled rows before the change, new = since."""
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
        """Dismiss the alert without changing the model; suppress new alerts for `snooze` batches.
        The detector restarts, so the current (elevated) error level becomes its new baseline."""
        n = self.cfg.snooze
        self.snooze_until = self.k + n
        self._log("Kept current model", detail or f"no alerts for {n} batches")
        self._clear_alert()
        self._reset_detector()
        return f"Kept the current model (v{self.version}). No new alerts for {n} batches."

    def apply(self, name):
        """Returns (ok, message). The deployed model, version and detector change only on success."""
        if name == KEEP:
            return True, self.keep()
        c, al = self.cand, self.alert
        try:
            if not c or name not in c.errs:
                raise ValueError("this option was not evaluated")
            data = self.get_data()
            model = build(self, name, data)                # final build on ALL new labelled rows
            model.predict(data[2].iloc[:1])                # smoke test before accepting
            det = make_monitor(self.cfg)                   # build the new detector before touching any state
        except Exception as exc:
            self._log("Update failed", f"{exc}. Deployed model unchanged.")
            return False, f"Update failed: {exc}. The deployed model was not changed."
        self.model, self.det = model, det
        self.version += 1
        self.version_start = self.k                        # batches before this were scored by the old model
        self.retrains.append(Retrain(version=self.version, strategy=name, applied=self.k, start=al.start))
        self._log("Model updated", f"v{self.version} trained on {c.n_new:,} new rows; drift detector reset")
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
