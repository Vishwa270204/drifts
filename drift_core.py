"""Core logic for the Model Drift Dashboard (no Streamlit imports, usable from scripts and tests).

A labelled dataset is streamed in batches through ONE deployed Hoeffding Adaptive Tree (`river`). Drift is detected
from per-batch error. When drift is confirmed, the user can apply ONE adaptation technique (online incremental
learning) or keep the current model:

    Strategy (registry)  ->  plan function (data selection)  ->  execute() (training procedure)

The candidate is trained WITHOUT the newest labelled rows and scored on those same held-out rows. The deployed model
only changes after `Session.apply` succeeds.

Typical use:
    sess = Session.create(df, target="y", tcol=None, task="classification", cfg=Config())
    sess.run(None)                       # until a decision is needed or data ends
    if sess.ready:
        sess.ensure_candidates()
        ok, msg = sess.apply(sess.cand["rec"])
"""
from __future__ import annotations

import copy
import io
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:                                   # reported by the UI
    river_tree, river_drift, RIVER_OK = None, None, False

warnings.filterwarnings("ignore", category=UserWarning, module="river")

# ------------------------------------------------------------------ constants
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MIN_FIT = 30       # minimum usable labelled rows to train a tree
DELTAS = [1e-3, 1e-5, 1e-7, 1e-9]
ADWIN_DELTAS = [0.1, 0.01, 0.002, 0.0005]

INC, KEEP = "Continue Learning (new data only)", "Keep Current Model"

METRICS = {
    "classification": ["Error rate", "1 - macro F1", "Fault-class miss rate (1 - recall)"],
    "regression": ["MAE", "RMSE"],
}
ROW_WISE = {"Error rate", "MAE"}   # metrics with a per-row loss (paired noise check possible)


@dataclass
class Config:
    """All user-tunable settings. `z` is the drift threshold multiplier (see SENS)."""
    train_pct: int = 40
    B: int = 100                    # batch size (rows)
    z: float = 3.0
    delay: int = 0                  # label delay (batches)
    min_rows: int = 300             # new labelled rows needed before asking
    snooze: int = 10                # batches without alerts after "keep"
    hold_frac: float = 0.25         # share of new labelled rows held out for evaluation
    metric: Optional[str] = None
    pos: Optional[str] = None       # fault (positive) class
    seed: int = 42
    grace: int = 200
    delta: float = 1e-7
    tau: float = 0.05
    adwin: float = 0.002
    win_thr: int = 300


# ------------------------------------------------------------------ model
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
    """A Hoeffding Adaptive Tree plus the small amount of preprocessing it needs (median imputation for numeric
    columns; categorical columns are passed as nominal attributes)."""

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
        """Initial tree trained in one pass, in the given (time) order."""
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
        """Keep learning incrementally on the existing tree, one row at a time."""
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


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def ref_stats(task, e):
    m, s = float(e.mean()), float(e.std())
    s = max(s, 0.1) if task == "classification" else s
    return m, max(s, 0.1 * m, 1e-9)


# ------------------------------------------------------------------ metrics
def evaluate(task, y, p, pos=None):
    y, p = np.asarray(y), np.asarray(p)
    if task == "regression":
        e = np.abs(y.astype(float) - p.astype(float))
        return dict(mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())))
    labels = np.unique(np.concatenate([y, p]))
    out = dict(err=float((y != p).mean()),
               macro_f1=float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
               precision=np.nan, recall=np.nan, f1=np.nan, n_pos=int((y == pos).sum()) if pos is not None else 0)
    if pos is not None and pos in labels:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[pos], average=None, zero_division=0)
        out["precision"], out["f1"] = float(pr[0]), float(f1[0])
        if out["n_pos"] > 0:
            out["recall"] = float(rc[0])
    return out


def selection_score(metric, m):
    if metric == "Error rate":
        return m["err"]
    if metric == "1 - macro F1":
        return 1.0 - m["macro_f1"]
    if metric.startswith("Fault-class miss rate"):
        return 1.0 - m["recall"] if np.isfinite(m["recall"]) else np.nan
    return m["mae"] if metric == "MAE" else m["rmse"]


def confusion(y, p):
    """(labels, confusion matrix) over all labels seen in y or p."""
    labels = list(np.unique(np.concatenate([np.asarray(y), np.asarray(p)])))
    return labels, confusion_matrix(y, p, labels=labels)


# ------------------------------------------------------------------ data prep
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
    """Stable string labels: integral numbers become '0','1' (not '0.0','1.0'). Expects no missing values."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


def guess_task(df, target):
    numeric = pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target])
    return "regression" if (numeric and df[target].nunique() > 10) else "classification"


def class_options(target_series):
    """(labels, default fault class). Default: '1' if present, else the rarest class."""
    ls = label_str(target_series.dropna())
    labs = sorted(set(ls))[:50]
    if not labs:
        return [], None
    rare = str(pd.Series(ls).value_counts().index[-1])
    return labs, "1" if "1" in labs else (rare if rare in labs else labs[0])


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


# ------------------------------------------------------------------ strategies: data selection -> training
@dataclass
class Plan:
    kind: str                       # keep | continue
    X: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None
    note: str = ""


@dataclass
class Strategy:
    name: str
    plan: Callable
    blurb: str


def plan_inc(sess, Xo, yo, Xn, yn):
    return Plan("continue", Xn, yn, note=f"copy of the deployed tree keeps learning from {len(yn):,} new rows; its "
                                         "ADWIN detectors can replace outdated branches")


def plan_keep(sess, Xo, yo, Xn, yn):
    return Plan("keep", note="deployed model unchanged")


STRATEGIES = [
    Strategy(INC, plan_inc, "Keep learning on a copy of the deployed tree using only the new rows, one row at a "
                            "time. The tree's own drift detectors swap out branches that stopped working."),
    Strategy(KEEP, plan_keep, "Leave the deployed model untouched."),
]
REG = {s.name: s for s in STRATEGIES}


def execute(sess, plan):
    """Training procedure for a plan. Never mutates the deployed model (continue plans use a deep copy)."""
    if plan.kind == "keep":
        return sess.model
    if plan.kind == "continue":
        return copy.deepcopy(sess.model).learn(plan.X, plan.y)
    raise ValueError(f"unknown plan kind {plan.kind}")


def build(sess, name, data):
    plan = REG[name].plan(sess, *data)
    return execute(sess, plan), plan.note


# ------------------------------------------------------------------ session
@dataclass
class Alert:
    start: int                      # batch index where the suspected drift began
    status: str = "waiting"         # waiting | ready


@dataclass
class Retrain:
    version: int
    strategy: str
    applied: int
    start: int
    score_new: float
    score_keep: float
    metric: str


@dataclass
class Session:
    """State and behaviour of one monitoring run. Contains no UI code."""
    X: pd.DataFrame
    y: np.ndarray
    task: str
    num: list
    cat: list
    cfg: Config
    n_tr: int
    N: int                          # rows in the streaming part
    nb: int                         # number of batches
    metric: str
    pos: Optional[str]
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
    cand: Optional[dict] = None     # result of compute_candidates (or {"error": ...})
    events: list = field(default_factory=list)
    retrains: list = field(default_factory=list)

    # ---- construction
    @classmethod
    def create(cls, df, target, tcol, task, cfg: Config) -> "Session":
        """Build a monitoring session. Raises ValueError with a readable message on bad input."""
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
            d, y = d.loc[ok], yv[ok].to_numpy(float)
            d = d.reset_index(drop=True)
        else:
            d = d.reset_index(drop=True)
            y = label_str(d[target])
        B = int(cfg.B)
        n_tr = int(len(d) * cfg.train_pct / 100)
        n_fit = int(n_tr * 0.8)
        if len(d) < 400:
            raise ValueError("Need at least 400 labelled rows.")
        if n_fit < 50 or len(d) - n_tr < 2 * B:
            raise ValueError("Initial training part is too small, or the streaming part has fewer than 2 batches. "
                             "Lower the batch size or adjust the training percentage.")
        num, cat = pick_features(d.iloc[:n_tr], {target, tcol})   # chosen from the initial training rows only
        if not (num or cat):
            raise ValueError("No usable feature columns were found.")
        X = build_frame(d, num, cat)
        pos = cfg.pos
        if task == "classification" and pos is None:
            vals, cnt = np.unique(y, return_counts=True)
            pos = str(vals[np.argmin(cnt)])
        metric = cfg.metric if cfg.metric in METRICS[task] else METRICS[task][0]
        try:
            model = Bundle(task, num, cat, cfg).fit(X.iloc[:n_fit], y[:n_fit])
        except Exception as exc:
            raise ValueError(f"Could not train the first model: {exc}")
        e = row_err(task, y[n_fit:n_tr], model.predict(X.iloc[n_fit:n_tr]))
        ref_m, ref_s = ref_stats(task, e)
        N = len(y) - n_tr
        return cls(X=X, y=y, task=task, num=num, cat=cat, cfg=cfg, n_tr=n_tr, N=N, nb=int(np.ceil(N / B)),
                   metric=metric, pos=pos, model=model, ref_m=ref_m, ref_s=ref_s)

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
        return "Drift suspected, collecting data" if self.alert else "Monitoring"

    @property
    def new_rows(self) -> int:
        """Labelled rows available since the suspected drift start (respects label delay)."""
        al = self.alert
        if al is None:
            return 0
        known = self.k - self.cfg.delay
        return max(0, min(known * self.cfg.B, self.N) - al.start * self.cfg.B)

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
                    self._log("Alert cleared", "error returned to normal before retraining was needed")
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

    # ---- candidate comparison and decisions
    def ensure_candidates(self):
        """Compute candidates once per alert; failures are stored as {"error": ...}."""
        if self.cand is None:
            try:
                self.cand = self.compute_candidates()
            except Exception as exc:
                self.cand = dict(error=f"Could not evaluate candidates: {exc}")
        return self.cand

    def compute_candidates(self):
        """Train the candidate WITHOUT the newest labelled rows and score it against the current model on them."""
        Xo, yo, Xn, yn = self.get_data()
        n = len(yn)
        if n < 2 * MIN_FIT:
            return dict(error=f"Only {n} new labelled rows: not enough to hold out an evaluation set and still "
                              f"train (need at least {2 * MIN_FIT}). Skip for now, or lower 'Min new labelled rows'.")
        hold = int(np.clip(round(n * self.cfg.hold_frac), MIN_FIT, n - MIN_FIT))
        Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
        task, pos = self.task, self.pos
        mets, errs, preds, notes, failed = {}, {}, {}, {}, []
        for sp in STRATEGIES:
            try:
                model, note = build(self, sp.name, (Xo, yo, Xt, yt))
                p = model.predict(Xh)
                if task == "regression" and not np.all(np.isfinite(np.asarray(p, float))):
                    raise ValueError("non-finite predictions")
                mets[sp.name] = evaluate(task, yh, p, pos)
                errs[sp.name], preds[sp.name], notes[sp.name] = row_err(task, yh, p), np.asarray(p), note
            except Exception as exc:
                failed.append((sp.name, f"{type(exc).__name__}: {exc}"))
        if KEEP not in mets:
            return dict(error="The current model could not be evaluated on the held-out rows.")
        metric, warns = self.metric, []
        if metric.startswith("Fault-class") and not np.isfinite(mets[KEEP]["recall"]):
            metric = "Error rate"
            warns.append(f"The held-out rows contain no '{pos}' (fault) examples, so recall is undefined. "
                         "Ranking uses error rate instead.")
        rows = []
        for sp in STRATEGIES:
            if sp.name not in mets:
                continue
            m = mets[sp.name]
            row = {"Strategy": sp.name, "Selection score": selection_score(metric, m)}
            if task == "regression":
                row.update({"MAE": m["mae"], "RMSE": m["rmse"]})
            else:
                row.update({"Error rate": m["err"], "Precision (fault class)": m["precision"],
                            "Recall (fault class)": m["recall"], "F1 (fault class)": m["f1"],
                            "Macro F1": m["macro_f1"]})
            rows.append(row)
        t = pd.DataFrame(rows)
        k_s = float(t.loc[t["Strategy"] == KEEP, "Selection score"].iloc[0])
        t["Better than keeping (%)"] = (k_s - t["Selection score"]) / k_s * 100 if k_s > 1e-12 else np.nan
        # recommendation: lowest score; ties (to 12 d.p.) go to Keep, then to registry order
        order = {sp.name: i for i, sp in enumerate(STRATEGIES)}
        valid = t[np.isfinite(t["Selection score"])].copy()
        valid["_s"] = valid["Selection score"].round(12)
        valid["_o"] = [0 if nme == KEEP else 1 + order[nme] for nme in valid["Strategy"]]
        rec = valid.sort_values(["_s", "_o"]).iloc[0]["Strategy"] if len(valid) else KEEP
        # reliability notes
        if hold < 100:
            warns.append(f"Only {hold} held-out rows: differences between strategies may be noise.")
        if task == "classification":
            npos = mets[KEEP]["n_pos"]
            if npos < 10:
                warns.append(f"Only {npos} '{pos}' (fault) rows in the held-out set: recall and F1 are unreliable.")
            share = max(np.mean(yh == v) for v in np.unique(yh))
            if share > 0.95 and metric == "Error rate":
                warns.append(f"The held-out set is {share:.0%} one class; error rate can look good for a model that "
                             "never predicts the fault class. Consider the fault-class miss rate in Advanced settings.")
        if rec != KEEP and metric in ROW_WISE:
            d = errs[KEEP] - errs[rec]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            if d.mean() <= se:
                warns.append(f"{rec} beats the current model by less than one standard error on these rows; "
                             "treat the difference as a tie.")
        r_s = float(t.loc[t["Strategy"] == rec, "Selection score"].iloc[0])
        if rec == KEEP:
            why = (f"**Recommended: {KEEP}.** The candidate did not have a lower held-out {metric} "
                   f"({k_s:.4f}) than the current model.")
        else:
            why = (f"**Recommended: {rec}.** It had the lowest held-out {metric} ({r_s:.4f} vs {k_s:.4f} for the "
                   f"current model) among candidates that ran successfully.")
        why += f" This is a measurement on {hold:,} recent labelled rows, not a guarantee for future data."
        return dict(table=t, errs=errs, preds=preds, yh=yh, notes=notes, hold=hold, n_old=len(yo), n_new=n,
                    n_train=len(yt), failed=failed, metric=metric, warnings=warns, rec=rec, why=why)

    def keep(self, detail: Optional[str] = None) -> str:
        """Dismiss the alert without changing the model; suppress new alerts for `snooze` batches."""
        n = self.cfg.snooze
        self.snooze_until = self.k + n
        self._log("Kept current model", detail or f"no alerts for {n} batches")
        self._clear_alert()
        return f"Kept the current model (v{self.version}). No new alerts for {n} batches."

    def apply(self, name):
        """Apply a decision. Returns (ok, message). The deployed model and version change only on success."""
        if name == KEEP:
            return True, self.keep()
        c, al = self.cand, self.alert
        try:
            if not c or name not in c.get("errs", {}):
                raise ValueError("this strategy has no valid evaluation")
            data = self.get_data()
            model, _ = build(self, name, data)             # final build on ALL new labelled rows
            model.predict(data[2].iloc[:1])                # smoke test before accepting
        except Exception as exc:
            self._log("Retraining failed", f"{name}: {exc}. Deployed model unchanged.")
            return False, f"Retraining with {name} failed: {exc}. The deployed model was not changed."
        self.model = model
        self.version += 1
        self.ref_m, self.ref_s = ref_stats(self.task, c["errs"][name])   # threshold from honest held-out errors
        t = c["table"].set_index("Strategy")["Selection score"]
        self.retrains.append(Retrain(version=self.version, strategy=name, applied=self.k, start=al.start,
                                     score_new=float(t[name]), score_keep=float(t[KEEP]), metric=c["metric"]))
        self._log("Retrained", f"v{self.version} = {name} (old {c['n_old']:,} rows, new {c['n_new']:,} rows)")
        self._clear_alert()
        return True, f"Deployed model v{self.version} ({name})."

    # ---- reporting
    @property
    def unit(self) -> str:
        return "MAE" if self.task == "regression" else "error rate"

    def retrain_summary(self) -> pd.DataFrame:
        """Live before/after comparison for every applied retraining."""
        ver, errs, unit = np.asarray(self.ver), np.asarray(self.errs, float), self.unit
        rows = []
        for r in self.retrains:
            idx = np.arange(r.start, min(r.applied, len(errs)))
            before = errs[idx][ver[idx] == r.version - 1] if len(idx) else np.array([])
            after = errs[ver == r.version]
            b, a = (before.mean() if len(before) else np.nan), (after.mean() if len(after) else np.nan)
            rows.append({"Version": f"v{r.version}", "Strategy": r.strategy, "Applied at batch": r.applied + 1,
                         f"Held-out {r.metric} at decision": r.score_new,
                         "Held-out score of old model": r.score_keep,
                         f"Live {unit}: old model since drift": b, f"Live {unit}: this model": a,
                         "Batches measured": int(len(after)),
                         "Live change (%)": (b - a) / b * 100 if np.isfinite(b) and b > 1e-12 and np.isfinite(a)
                         else np.nan})
        return pd.DataFrame(rows)
