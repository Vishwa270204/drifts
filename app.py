"""HAT industrial-IoT failure prediction - single-file Streamlit app.

Run:  pip install -r requirements.txt && streamlit run streamlit_app.py

Model: ONLY river.tree.HoeffdingAdaptiveTreeClassifier (HAT). No RF / ARF.
Flow : predict on each record -> wait for the VERIFIED label -> evaluate the stored prediction
       -> learn_one(stored features, label)      (prequential: predict, evaluate, then learn)
DATA : synthetic + simulated stream. Scores demonstrate the pipeline; they are NOT evidence of
       real industrial accuracy, and nothing here is connected to real machinery.
"""
from __future__ import annotations

import hashlib, heapq, itertools, logging, math, os, pickle, tempfile, threading, time, uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import river
import streamlit as st
from river import tree

log = logging.getLogger("iot_hat")

# ============================================================== 1. CONFIG
FEATURES = ("temperature", "vibration", "current", "voltage", "load")   # the ONLY model inputs
RANGES = {"temperature": (-20, 200), "vibration": (0, 50), "current": (0, 500),
          "voltage": (0, 1000), "load": (0, 1.5)}                          # plausibility limits
_e = os.environ.get
CFG = dict(
    artifact_dir=Path(_e("IOT_HAT_ARTIFACT_DIR", "artifacts")),
    seed=int(_e("IOT_HAT_SEED", 42)), historical_rows=12000, train_fraction=0.8,
    rolling_window=int(_e("IOT_HAT_ROLLING_WINDOW", 200)), min_samples=int(_e("IOT_HAT_MIN_SAMPLES", 20)),
    checkpoint_every=int(_e("IOT_HAT_CHECKPOINT_EVERY", 200)), keep_checkpoints=5, max_pending=5000,
    label_delay=int(_e("IOT_HAT_LABEL_DELAY", 8)), label_drop_rate=float(_e("IOT_HAT_LABEL_DROP_RATE", 0.02)),
    hat=dict(grace_period=100, delta=1e-5, leaf_prediction="nba", drift_window_threshold=300,
             switch_significance=0.05, seed=int(_e("IOT_HAT_SEED", 42))),
)

# ============================================================== 2. SYNTHETIC DATA
REGIME_BLOCK, CYCLE = 400, 3000
COV_START, COV_END, CONCEPT_AT = 8000, 10000, 14000      # covariate-drift ramp; concept drift index
BASE_TIME, STEP_S = pd.Timestamp("2025-01-01", tz="UTC"), 30


def generate(seed: int, start: int, n: int) -> pd.DataFrame:
    """Records start..start+n-1 on a global timeline. regime/concept columns are diagnostics only."""
    idx = np.arange(start, start + n)
    rng = np.random.default_rng([seed, start, n])
    blocks = idx // REGIME_BLOCK
    reg_of = {b: int(np.random.default_rng([seed, 9, int(b)]).choice(3, p=[.25, .5, .25])) for b in np.unique(blocks)}
    regime = np.array([reg_of[b] for b in blocks])
    load = np.clip(np.array([.25, .60, .90])[regime] + 0.07 * rng.standard_normal(n), 0.02, 1.2)
    wear = (idx % CYCLE) / CYCLE                                     # hidden maintenance-cycle wear
    ramp = np.clip((idx - COV_START) / (COV_END - COV_START), 0, 1)  # covariate drift (inputs shift)
    ambient = 22 + 5 * np.sin(2 * np.pi * idx / 2880) + 6 * ramp
    temp = ambient + 32 * load + 14 * wear + rng.normal(0, 1.2, n)
    vib = 1 + 2 * load + 2.8 * wear ** 2 + 0.6 * ramp + rng.normal(0, .15, n)
    cur = 5 + 40 * load * (1 + .12 * wear) + rng.normal(0, .8, n)
    volt = 400 - 8 * load + rng.normal(0, 1.5, n)
    drift = idx >= CONCEPT_AT                                         # concept drift: P(y|x) changes
    stress = np.where(drift, .30 * wear + .85 * load, .80 * wear + .35 * load)
    p = 1 / (1 + np.exp(-14 * (stress - np.where(drift, .92, .96))))
    return pd.DataFrame({"timestamp": BASE_TIME + pd.to_timedelta(idx * STEP_S, unit="s"),
                         "temperature": temp, "vibration": vib, "current": cur, "voltage": volt, "load": load,
                         "failure": (rng.random(n) < p).astype(int), "regime": regime, "concept": drift.astype(int)})


# ============================================================== 3. VALIDATION / PREPROCESSING
class Bad(ValueError):
    """Invalid record or label."""


def parse_ts(v):
    try:
        t = pd.Timestamp(v)
        if pd.isna(t):
            raise ValueError
    except Exception:
        raise Bad(f"invalid timestamp: {v!r}")
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()


def parse_label(v) -> int:
    try:
        f = float(v.strip() if isinstance(v, str) else v)
    except (TypeError, ValueError):
        raise Bad(f"missing/non-numeric label: {v!r}")
    if math.isnan(f) or f not in (0.0, 1.0):
        raise Bad(f"non-binary label: {v!r}")
    return int(f)


class Pre:
    """Frozen z-score fitted on the TRAIN split only; stored in the artifact."""

    def __init__(self, means, stds):
        self.means, self.stds = means, {k: (v or 1.0) for k, v in stds.items()}

    @classmethod
    def fit(cls, df):
        return cls(df[list(FEATURES)].mean().to_dict(), df[list(FEATURES)].std(ddof=0).to_dict())

    def validate(self, rec) -> dict:
        out = {}
        for f in FEATURES:
            if rec.get(f) is None:
                raise Bad(f"missing sensor value: {f}")
            try:
                v = float(rec[f])
            except (TypeError, ValueError):
                raise Bad(f"non-numeric {f}: {rec[f]!r}")
            if not math.isfinite(v) or not RANGES[f][0] <= v <= RANGES[f][1]:
                raise Bad(f"{f}={v} not finite / outside plausible range {RANGES[f]}")
            out[f] = v
        return out

    def transform(self, raw):
        return {f: (raw[f] - self.means[f]) / self.stds[f] for f in FEATURES}


# ============================================================== 4. METRICS (prequential)
class Confusion:
    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0

    @property
    def n(self):
        return self.tp + self.fp + self.tn + self.fn

    def add(self, t, p, s=1):
        if t and p: self.tp += s
        elif not t and p: self.fp += s
        elif not t and not p: self.tn += s
        else: self.fn += s

    def metrics(self, min_n=1):
        """Undefined -> None (never 0): needs >= min_n samples; precision needs predicted positives,
        recall needs actual positives, F1 needs both."""
        m = dict(n=self.n, tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn,
                 accuracy=None, precision=None, recall=None, f1=None)
        if self.n < max(1, min_n):
            return m
        m["accuracy"] = (self.tp + self.tn) / self.n
        if self.tp + self.fp: m["precision"] = self.tp / (self.tp + self.fp)
        if self.tp + self.fn: m["recall"] = self.tp / (self.tp + self.fn)
        p, r = m["precision"], m["recall"]
        if p is not None and r is not None:
            m["f1"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return m


class Perf:
    def __init__(self, window, min_n, hist=5000):
        self.window, self.min_n = window, min_n
        self.cum, self.roll, self.buf, self.hist = Confusion(), Confusion(), deque(), deque(maxlen=hist)

    def record(self, t, p, rid, ts):
        self.cum.add(t, p); self.roll.add(t, p); self.buf.append((t, p))
        if len(self.buf) > self.window:
            self.roll.add(*self.buf.popleft(), s=-1)
        pt = dict(n=self.cum.n, record_id=rid, timestamp=ts, rolling=self.roll.metrics(self.min_n),
                  cumulative=self.cum.metrics(self.min_n))
        self.hist.append(pt)
        return pt

    def snapshot(self):
        return dict(rolling=self.roll.metrics(self.min_n), cumulative=self.cum.metrics(self.min_n),
                    window=self.window, min_samples=self.min_n)


# ============================================================== 5. PERSISTENCE (atomic + checksummed)
def _atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp): os.unlink(tmp)
        raise


def save_artifact(path: Path, model, meta: dict):
    blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    payload = dict(meta=dict(meta, saved_at=datetime.now(timezone.utc).isoformat()), blob=blob,
                   sha256=hashlib.sha256(blob).hexdigest())
    _atomic(path, pickle.dumps(payload))                 # ONE file: model + metadata + checksum
    log.info("saved model artifact %s", path)


def load_latest(root: Path):
    """Newest valid artifact: checkpoints (newest first) then initial. Corrupt / feature-mismatched
    files are skipped; if none is valid -> RuntimeError. An untrained model is never returned."""
    cands = sorted((root / "checkpoints").glob("ckpt_*.pkl"), reverse=True) + [root / "initial.pkl"]
    cands = [c for c in cands if c.exists()]
    if not cands:
        raise RuntimeError(f"no trained HAT artifact under {root}")
    errs = []
    for p in cands:
        try:
            d = pickle.loads(p.read_bytes())
            if hashlib.sha256(d["blob"]).hexdigest() != d["sha256"]:
                raise ValueError("checksum mismatch")
            if list(d["meta"]["features"]) != list(FEATURES):
                raise ValueError("feature mismatch")
            model = pickle.loads(d["blob"])
            assert hasattr(model, "learn_one") and hasattr(model, "predict_proba_one")
            log.info("loaded model artifact %s", p)
            return model, d["meta"], p
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping invalid artifact %s: %s", p, exc); errs.append(f"{p.name}: {exc}")
    raise RuntimeError("all artifacts invalid: " + "; ".join(errs))


# ============================================================== 6. OFFLINE TRAINING
def train_initial(cfg=CFG, df: pd.DataFrame | None = None) -> dict:
    """Chronological split -> scaler fitted on train only -> HAT.learn_one over train -> score holdout."""
    df = generate(cfg["seed"], 0, cfg["historical_rows"]) if df is None else df
    df = df.dropna(subset=[*FEATURES, "failure", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * cfg["train_fraction"])
    tr, ho = df.iloc[:cut], df.iloc[cut:]
    assert tr.timestamp.max() <= ho.timestamp.min()
    pre = Pre.fit(tr)
    model = tree.HoeffdingAdaptiveTreeClassifier(**cfg["hat"])
    for row in tr[list(FEATURES) + ["failure"]].itertuples(index=False):
        model.learn_one(pre.transform(dict(zip(FEATURES, row[:-1]))), int(row[-1]))
    cm = Confusion()
    for row in ho[list(FEATURES) + ["failure"]].itertuples(index=False):      # holdout: never learned
        cm.add(int(row[-1]), int(model.predict_one(pre.transform(dict(zip(FEATURES, row[:-1]))))))
    meta = dict(model_id=f"hat-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
                features=list(FEATURES), means=pre.means, stds=pre.stds, river_version=river.__version__,
                n_updates_total=len(tr), n_online_updates=0, train_rows=len(tr), holdout_rows=len(ho),
                holdout_metrics=cm.metrics(), kind="initial",
                provenance="SYNTHETIC demonstration data; not evidence of real-world accuracy")
    save_artifact(cfg["artifact_dir"] / "initial.pkl", model, meta)
    return meta


# ============================================================== 7. PREQUENTIAL ENGINE
DRIFT_LABELS = {"n_alternate_trees": "Drift signalled: alternate subtree started",
                "n_pruned_alternate_trees": "Alternate subtree pruned (no improvement)",
                "n_switch_alternate_trees": "Adaptation: alternate subtree replaced original"}


class Engine:
    """predict (ingest) -> evaluate -> learn (submit_label). Drift events = real HAT counter changes."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.model, self.meta, self.src_path = load_latest(cfg["artifact_dir"])
        self.pre = Pre(self.meta["means"], self.meta["stds"])
        self.updates_total = int(self.meta["n_updates_total"])
        self.pending: "OrderedDict[str, dict]" = OrderedDict()
        self.final, self.final_q = set(), deque()
        self.expired = 0
        self.perf = Perf(cfg["rolling_window"], cfg["min_samples"])
        self.c = dict(processed=0, rejected=0, labels=0, bad_labels=0, dup_labels=0, unknown_labels=0,
                      evaluated=0, learned=0)
        self.latest_pred = self.latest_learned = self.last_error = None
        self.events: deque = deque(maxlen=100)
        self._ids = itertools.count(1)
        self._ctr = self.counters()
        self.since_ckpt = 0
        self.ckpt = dict(path=str(self.src_path) if self.meta.get("kind") == "checkpoint" else None,
                         at=None, error=None, count=0)

    version = property(lambda s: f"{s.meta['model_id']}+u{s.updates_total}")

    def counters(self):
        m = self.model
        return {k: int(getattr(m, k)) for k in DRIFT_LABELS}

    def ingest(self, rec: dict) -> dict:
        try:
            ts = parse_ts(rec.get("timestamp"))
            raw = self.pre.validate(rec)
            rid = str(rec.get("record_id") or f"r{next(self._ids):08d}")
            if rid in self.pending or rid in self.final:
                raise Bad(f"duplicate record_id {rid}")
        except Bad as exc:
            self.c["rejected"] += 1; self.last_error = str(exc); raise
        x = self.pre.transform(raw)
        proba = self.model.predict_proba_one(x)
        p1 = float(proba.get(1, 0.0)) if proba else None
        pred = int(max(proba, key=proba.get)) if proba else int(self.model.predict_one(x))
        self.c["processed"] += 1
        self.pending[rid] = dict(x=x, pred=pred, ts=ts, version=self.version)
        while len(self.pending) > self.cfg["max_pending"]:         # label never arrived: expire, never learn
            old, _ = self.pending.popitem(last=False); self._finalize(old); self.expired += 1
        self.latest_pred = dict(record_id=rid, timestamp=ts, sensors=raw, prediction=pred, probability=p1,
                                model_version=self.version, n=self.c["processed"], status="awaiting verified label")
        return self.latest_pred

    def _finalize(self, rid):
        self.final.add(rid); self.final_q.append(rid)
        while len(self.final_q) > 100_000:
            self.final.discard(self.final_q.popleft())

    def submit_label(self, rid, label) -> dict:
        self.c["labels"] += 1
        rid = str(rid)
        try:
            y = parse_label(label)
        except Bad as exc:                                          # record stays pending
            self.c["bad_labels"] += 1; self.last_error = str(exc)
            return dict(status="invalid_label", record_id=rid)
        if rid in self.final:
            self.c["dup_labels"] += 1; return dict(status="duplicate_or_expired", record_id=rid)
        p = self.pending.pop(rid, None)                             # single gate: at-most-once
        if p is None:
            self.c["unknown_labels"] += 1; return dict(status="unknown_record", record_id=rid)
        self._finalize(rid)
        pt = self.perf.record(y, p["pred"], rid, p["ts"])           # 1) EVALUATE stored (pre-learning) prediction
        self.c["evaluated"] += 1
        self.model.learn_one(p["x"], y)                             # 2) LEARN (right or wrong) on stored vector
        self.updates_total += 1; self.since_ckpt += 1; self.c["learned"] += 1
        self.latest_learned = dict(record_id=rid, label=y, predicted=p["pred"], correct=y == p["pred"],
                                   predicted_by=p["version"], learned_into=self.version)
        new = self.counters()
        for k, v in new.items():                                    # genuine HAT events only
            if v > self._ctr[k]:
                self.events.append(dict(event=DRIFT_LABELS[k], count=v - self._ctr[k], at_update=self.c["learned"],
                                        record_id=rid, nodes=int(self.model.n_nodes),
                                        time=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self._ctr = new
        every = self.cfg["checkpoint_every"]
        if every > 0 and self.since_ckpt >= every:
            self.checkpoint("periodic")
        return dict(status="learned", record_id=rid, evaluation=pt)

    def checkpoint(self, reason="manual"):
        try:
            meta = dict(self.meta, kind="checkpoint", n_updates_total=self.updates_total,
                        n_online_updates=self.c["learned"], counters=self._ctr, reason=reason,
                        river_version=river.__version__)
            path = self.cfg["artifact_dir"] / "checkpoints" / f"ckpt_{self.updates_total:010d}.pkl"
            save_artifact(path, self.model, meta)
            for old in sorted(path.parent.glob("ckpt_*.pkl"))[:-self.cfg["keep_checkpoints"]]:
                old.unlink(missing_ok=True)
            self.since_ckpt = 0
            self.ckpt = dict(path=str(path), at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             error=None, count=self.ckpt["count"] + 1)
            return path
        except Exception as exc:  # noqa: BLE001
            log.exception("checkpoint failed"); self.ckpt["error"] = f"{type(exc).__name__}: {exc}"
            return None


# ============================================================== 8. SIMULATED LIVE STREAM + RUNTIME
class Simulator:
    """Continues the synthetic timeline after the history. Labels arrive `delay` records later; some never do.
    The hidden label is never part of the emitted record."""

    def __init__(self, seed, delay, drop, start=12000, chunk=256):
        self.seed, self.idx, self.delay, self.drop, self.chunk = seed, start, delay, drop, chunk
        self.rng, self.buf, self.due, self.tie, self.n = np.random.default_rng(seed + 1), [], [], itertools.count(), 0

    def next(self):
        if not self.buf:
            self.buf = generate(self.seed, self.idx, self.chunk).to_dict("records")[::-1]
        row = self.buf.pop(); self.idx += 1; self.n += 1
        rid = f"sim{self.n:08d}"
        if self.rng.random() >= self.drop:
            heapq.heappush(self.due, (self.n + self.delay, next(self.tie), rid, int(row["failure"])))
        return dict(record_id=rid, timestamp=datetime.now(timezone.utc).isoformat(), **{f: float(row[f]) for f in FEATURES})

    def labels(self):
        out = []
        while self.due and self.due[0][0] <= self.n:
            _, _, rid, y = heapq.heappop(self.due); out.append((rid, y))
        return out


class Runtime:
    """One shared engine + stream, advanced by wall-clock time on every dashboard refresh."""

    def __init__(self):
        self.lock, self.running, self.rate, self.last = threading.RLock(), True, 4.0, time.monotonic()
        self.error, self.engine, self.sim = None, None, None
        self.recent = deque(maxlen=60)
        try:
            self.engine = Engine(CFG)                    # raises if no valid artifact: never an untrained fallback
            self.sim = Simulator(CFG["seed"], CFG["label_delay"], CFG["label_drop_rate"])
        except RuntimeError as exc:
            self.error = str(exc)

    def advance(self, now=None, max_steps=60) -> int:
        if not self.engine:
            return 0
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.running:
                self.last = now; return 0
            due = int((now - self.last) * self.rate); n = min(due, max_steps)
            self.last = now if due > max_steps else self.last + n / self.rate
            for _ in range(n):
                try:
                    self.recent.append(self.engine.ingest(self.sim.next()))
                except Bad as exc:
                    log.warning("rejected record: %s", exc)
                for rid, y in self.sim.labels():
                    self.engine.submit_label(rid, y)
            return n


# ============================================================== 9. UI
def metric_fig(hist, metric, window):
    pts = list(hist)[-3000:]
    x = [p["n"] for p in pts]
    f = go.Figure()
    f.add_scatter(x=x, y=[p["rolling"][metric] for p in pts], name=f"rolling (last {window})", connectgaps=False)
    f.add_scatter(x=x, y=[p["cumulative"][metric] for p in pts], name="cumulative", connectgaps=False, line=dict(dash="dot"))
    f.update_layout(title=f"{metric.upper() if metric == 'f1' else metric.title()} vs evaluated record # (verified labels only)",
                    xaxis_title="evaluated record number", yaxis=dict(range=[0, 1.02]), height=300,
                    margin=dict(l=40, r=10, t=45, b=40), legend=dict(orientation="h", y=-0.3))
    if not any(p["rolling"][metric] is not None or p["cumulative"][metric] is not None for p in pts):
        f.add_annotation(text="No verified labels evaluated yet" if not pts else "Undefined so far (too few samples / zero denominator)",
                         xref="paper", yref="paper", x=.5, y=.5, showarrow=False)
    return f


def fmt(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


@st.cache_resource(show_spinner="Loading trained HAT artifact…")
def get_runtime() -> Runtime:
    return Runtime()


def main():
    st.set_page_config(page_title="HAT Industrial IoT Failure Prediction", page_icon="🏭", layout="wide")
    logging.basicConfig(level=logging.INFO)
    rt = get_runtime()
    st.title("🏭 Industrial IoT Failure Prediction")
    st.caption("Hoeffding Adaptive Tree (River) · prequential online learning · SYNTHETIC data, SIMULATED stream")

    if not rt.engine:                                    # explicit action required; never an untrained model
        st.error("**MODEL UNAVAILABLE** – no valid trained HAT artifact. The stream is not running.")
        st.code(rt.error or "")
        if st.button("Train initial HAT now (offline, synthetic data)", type="primary"):
            with st.spinner("Training…"):
                train_initial()
            get_runtime.clear(); st.rerun()
        st.stop()

    with st.sidebar:
        st.header("Controls")
        run = st.toggle("Stream running", value=rt.running)
        if run != rt.running: rt.running, rt.last = run, time.monotonic()
        rate = st.slider("Simulated records / second", 0.5, 20.0, float(rt.rate), 0.5)
        rt.rate = rate
        refresh = st.slider("Dashboard refresh (s)", 1, 10, 2)
        if st.button("Checkpoint model now"):
            p = rt.engine.checkpoint()
            st.success(f"Saved {p}") if p else st.error(rt.engine.ckpt["error"])
        st.caption("Scores are demonstrations on synthetic data, not real-world accuracy.")

    def live():
        rt.advance()
        with rt.lock:
            e = rt.engine; c = e.c; mt = e.perf.snapshot(); hist = list(e.perf.hist); recent = list(rt.recent)
            lp, ll, evs, tot = e.latest_pred, e.latest_learned, list(e.events)[-20:], e.counters()
            ck, ver, nodes = dict(e.ckpt), e.version, (int(e.model.n_nodes), int(e.model.n_leaves), int(e.model.height))
            pending, expired, since, err = len(e.pending), e.expired, e.since_ckpt, e.last_error
        st.warning("⚠ SIMULATED stream – not connected to real machinery. Metrics describe simulated data only.")
        if err: st.info(f"Last rejected input: {err}")
        l, r = st.columns(2)
        with l:
            st.subheader("Live sensor readings")
            if not lp: st.info("Waiting for the first record…")
            else:
                for col, (k, u) in zip(st.columns(5), [("temperature", "°C"), ("vibration", "mm/s"), ("current", "A"),
                                                       ("voltage", "V"), ("load", "×rated")]):
                    col.metric(f"{k.title()} ({u})", f"{lp['sensors'][k]:.2f}")
                st.caption(f"record {lp['record_id']} · {lp['timestamp']}")
                if len(recent) > 1:
                    st.line_chart(pd.DataFrame([{**q["sensors"], "#": q["n"]} for q in recent]).set_index("#")[["temperature", "vibration"]], height=160)
        with r:
            st.subheader("Latest failure prediction")
            if not lp: st.info("No data")
            else:
                (st.error if lp["prediction"] else st.success)("FAILURE PREDICTED" if lp["prediction"] else "No failure predicted")
                st.caption(f"P(failure)={fmt(lp['probability'])} · model {lp['model_version']} · label: {lp['status']}")
            if ll:
                st.caption(f"Latest learned: {ll['record_id']} · true={ll['label']} · predicted={ll['predicted']} "
                           f"({'correct' if ll['correct'] else 'WRONG'}) · {ll['predicted_by']} → {ll['learned_into']}")
        for col, (lab, v) in zip(st.columns(8), [("Processed", c["processed"]), ("Labels received", c["labels"]),
                ("Evaluated", c["evaluated"]), ("Learning updates", c["learned"]), ("Awaiting label", pending),
                ("Rejected", c["rejected"]), ("Bad / dup labels", f"{c['bad_labels']} / {c['dup_labels']}"), ("Expired", expired)]):
            col.metric(lab, v)

        st.subheader("Performance (predicted before learning, scored on verified labels)")
        rows = [{"Window": n, "Samples": m["n"], "Accuracy": fmt(m["accuracy"]), "Precision": fmt(m["precision"]),
                 "Recall": fmt(m["recall"]), "F1": fmt(m["f1"]), "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"]}
                for n, m in ((f"Rolling (last {mt['window']})", mt["rolling"]), ("Cumulative", mt["cumulative"]))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"n/a = undefined (fewer than {mt['min_samples']} samples or a zero denominator); shown as gaps in graphs, never 0.")
        g = st.columns(2)
        for i, m in enumerate(("accuracy", "precision", "recall", "f1")):
            g[i % 2].plotly_chart(metric_fig(hist, m, mt["window"]), width="stretch", key=f"plot-{m}")

        a, b = st.columns(2)
        h = e.meta.get("holdout_metrics") or {}
        with a:
            st.subheader("HAT model status")
            st.table(pd.Series({"Status": f"loaded · {'streaming' if rt.running else 'paused'}", "Version": ver,
                                "Loaded from": f"{e.src_path} ({e.meta.get('kind')})", "River": river.__version__,
                                "Tree": f"{nodes[0]} nodes / {nodes[1]} leaves / height {nodes[2]}",
                                "Initial holdout (chronological)": f"acc {fmt(h.get('accuracy'))} · P {fmt(h.get('precision'))} · "
                                                                   f"R {fmt(h.get('recall'))} · F1 {fmt(h.get('f1'))}"}, name="value"))
        with b:
            st.subheader("Checkpoint & persistence")
            st.table(pd.Series({"Last checkpoint": ck["path"] or "none yet", "Saved at": ck["at"] or "–",
                                "Checkpoints this run": ck["count"], "Policy": f"every {CFG['checkpoint_every']} updates",
                                "Updates since checkpoint": since, "Last error": ck["error"] or "none"}, name="value").astype(str))
            st.caption("Hosted disks are often ephemeral: checkpoints may not survive a restart/redeploy.")
        st.subheader("HAT drift / adaptation events")
        st.caption(f"HAT counters (cumulative, incl. offline training): alternate subtrees started={tot['n_alternate_trees']} · "
                   f"pruned={tot['n_pruned_alternate_trees']} · switched={tot['n_switch_alternate_trees']}")
        if evs:
            st.dataframe(pd.DataFrame(evs[::-1]), hide_index=True, width="stretch")
        else:
            st.info("No drift/adaptation events observed since this model was loaded. Events come only from River's HAT "
                    "counters; sensor drift alone is not reported.")

    st.fragment(run_every=refresh if rt.running else None)(live)()


if __name__ == "__main__":
    main()
