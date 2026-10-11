"""IoT Model Drift Detection & Retraining Dashboard (single page, CSV-driven)."""
import copy
import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from river.drift import ADWIN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="IoT Drift & Retraining Dashboard", layout="wide")

SEED, EPS, MAX_CAT = 42, 1e-4, 30
INC, ONL, CON, TRA, FULL = ("Incremental learning", "Online learning", "Continual learning (replay buffer)",
                            "Transfer learning", "Full retraining")
METHODS = [INC, ONL, CON, TRA, FULL]
STRATS = ["New data only", "Old data + new data", "Sliding window"]
MODELS = {  # partial = supports partial_fit; transfer = weights can be carried over
    "SGD classifier (log-loss)": dict(partial=True, transfer=True),
    "MLP neural network": dict(partial=True, transfer=True),
    "Logistic Regression": dict(partial=False, transfer=True),
    "Random Forest": dict(partial=False, transfer=False),
}
S = st.session_state


# ───────────────────────── helpers ─────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    for c in df.columns:  # numeric-looking text columns -> numeric
        if df[c].dtype == object:
            n, nn = pd.to_numeric(df[c], errors="coerce"), df[c].notna().sum()
            if nn and n.notna().sum() >= 0.95 * nn:
                df[c] = n
    return df


def datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != object:
        return False
    smp = s.dropna().astype(str).head(50)
    return len(smp) > 0 and pd.to_datetime(smp, errors="coerce").notna().mean() >= 0.9


def check_features(df, feats, target, ts):
    num, cat, issues = [], [], []
    for c in feats:
        s = df[c]
        if c in (target, ts):
            issues.append(f"`{c}` is the target/timestamp column and was excluded from features.")
        elif s.notna().sum() == 0:
            issues.append(f"`{c}` is entirely missing and was excluded.")
        elif s.nunique(dropna=True) <= 1:
            issues.append(f"`{c}` is constant, so it carries no signal and was excluded.")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif datetime_like(s):
            issues.append(f"`{c}` looks like a date/time column; choose it as the timestamp column instead. Excluded.")
        elif s.nunique() > MAX_CAT:
            issues.append(f"`{c}` is text with {s.nunique()} distinct values (> {MAX_CAT}); one-hot encoding is "
                          "incompatible with every available model at that size. Excluded.")
        else:
            cat.append(c)
    return num, cat, issues


def make_model(name):
    if name.startswith("SGD"):
        return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=SEED)
    if name.startswith("MLP"):
        return MLPClassifier((32,), max_iter=300, random_state=SEED)
    if name.startswith("Logistic"):
        return LogisticRegression(max_iter=500)
    return RandomForestClassifier(150, n_jobs=-1, random_state=SEED)


def metrics(model, X, y, k):
    p = model.predict(X)
    kw = dict(average="binary", pos_label=1, zero_division=0) if k == 2 else dict(average="macro", zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, **kw)
    return dict(Accuracy=float((p == y).mean()), Precision=float(pr), Recall=float(rc), F1=float(f1),
                cm=confusion_matrix(y, p, labels=np.arange(k)))


def psi_vec(ref, cur):
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def col_props(spec, s):
    s = s.dropna()
    if len(s) < 10:
        return None
    if spec["kind"] == "num":
        idx, k = np.searchsorted(spec["edges"], s.to_numpy(float), side="right"), len(spec["edges"]) + 1
    else:
        idx, k = s.map(spec["cats"]).fillna(spec["n"]).astype(int).to_numpy(), spec["n"] + 1
    c = np.bincount(idx, minlength=k).astype(float)
    return c / c.sum()


def make_ref(s, kind, bins):
    if kind == "num":  # fixed reference bins from the baseline training data
        spec = dict(kind="num", edges=np.unique(np.quantile(s.dropna().to_numpy(float), np.linspace(0, 1, bins + 1)[1:-1])))
    else:
        cats = list(s.value_counts().index[:20])
        spec = dict(kind="cat", cats={c: i for i, c in enumerate(cats)}, n=len(cats))
    spec["props"] = col_props(spec, s)
    return spec


# ───────────────────────── pipeline preparation ─────────────────────────
def prepare(df, target, num, cat, ts, model_name, base_frac, hist_frac, delay, bins, win, thr, delta):
    feats = num + cat
    d = df[feats + [target] + ([ts] if ts else [])].copy()
    d = d[d[target].notna()]
    if ts:
        t = d[ts] if pd.api.types.is_numeric_dtype(d[ts]) else pd.to_datetime(d[ts], errors="coerce")
        d["_ts"] = t
        d = d[d["_ts"].notna()].sort_values("_ts", kind="stable")
    d = d.reset_index(drop=True)
    if not ts:
        d["_ts"] = d.index
    for c in cat:
        s = d[c]
        d[c] = s.astype(str).where(s.notna(), np.nan)
    names = sorted(d[target].unique(), key=str)
    if not 2 <= len(names) <= 20:
        raise ValueError(f"Target has {len(names)} distinct values; classification needs 2–20 classes. "
                         "Bin a continuous target or pick another column.")
    d["_y"] = d[target].map({v: i for i, v in enumerate(names)}).astype(int)
    n = len(d)
    nb = int(n * base_frac)
    ntr = int(nb * (1 - hist_frac))
    if ntr < 50 or nb - ntr < 20 or n - nb < 100:
        raise ValueError("Not enough rows for the chosen split (need ≥50 baseline-train, ≥20 historical hold-out, "
                         "≥100 post-baseline rows). Adjust the fractions or upload more data.")
    train, hist, rest = d.iloc[:ntr], d.iloc[ntr:nb], d.iloc[nb:]
    m = np.arange(len(rest)) % 5 == 4  # every 5th post-baseline row is held out (never replayed / trained on)
    recent, stream = rest[m], rest[~m]
    k = len(names)
    if train["_y"].nunique() < k:
        raise ValueError("The baseline-train split does not contain every class. Increase the baseline fraction, "
                         "merge rare classes, or pick another target.")
    tf = []
    if num:
        tf.append(("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num))
    if cat:
        tf.append(("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                  ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    pre = ColumnTransformer(tf)
    Xtr = pre.fit_transform(train[feats])
    ytr = train["_y"].to_numpy()
    model = make_model(model_name)
    model.fit(Xtr, ytr)
    Xh, yh = pre.transform(hist[feats]), hist["_y"].to_numpy()
    Xr, yr = pre.transform(recent[feats]), recent["_y"].to_numpy()
    F = stream[feats].reset_index(drop=True)
    anom = np.zeros(len(F), bool)
    if num:
        mu, sd = train[num].mean(), train[num].std().replace(0, 1)
        anom = ((F[num] - mu) / sd).abs().fillna(0).max(axis=1).to_numpy() > 4
    ref = {c: make_ref(train[c], "num" if c in num else "cat", bins) for c in feats}
    lab_cnt = np.bincount(ytr, minlength=k).astype(float)
    ys = stream["_y"].to_numpy()
    base_hist = metrics(model, Xh, yh, k)
    S.update(
        ready=True, sig=None, model=model, base_model=copy.deepcopy(model), model_name=model_name,
        names=[str(v) for v in names], feats=feats, num=num, cat=cat, target=target,
        Xtr=Xtr, ytr=ytr, Xh=Xh, yh=yh, Xr=Xr, yr=yr, Xs=pre.transform(stream[feats]), ys=ys, F=F,
        T=stream["_ts"].tolist(), anom=anom, ref=ref, label_ref=lab_cnt / lab_cnt.sum(),
        q_miss=train[feats].isna().mean(), q_lo=train[num].min() if num else {}, q_hi=train[num].max() if num else {},
        ref_err=1 - base_hist["Accuracy"], base_metrics=dict(hist=base_hist, recent=metrics(model, Xr, yr, k)),
        cfg=dict(delay=delay, win=win, thr=thr, delta=delta, k=k, has_ts=bool(ts)),
        counts=dict(train=len(train), hist=len(hist), recent=len(recent), stream=len(stream)),
        version=0, model_label="Baseline (v0)", decisions=[], cand=None,
    )
    reset_stream()


def reset_stream():
    n = len(S.ys)
    S.model, S.version, S.model_label = copy.deepcopy(S.base_model), 0, "Baseline (v0)"
    S.update(pos=0, lp=0, running=False, started=False, pred=np.full(n, -1), ver=np.zeros(n, int),
             err=np.full(n, np.nan), adwin=ADWIN(delta=S.cfg["delta"]), events=[], psi_hist=[], psis={},
             lpsi=0.0, psi_on=False, cand=None, decisions=[])


# ───────────────────────── drift logic ─────────────────────────
def quality_flags(win):
    fl = []
    d = (win[S.feats].isna().mean() - S.q_miss).clip(lower=0)
    if (d > 0.1).any():
        fl.append("missing readings rose in " + ", ".join(d[d > 0.1].index[:4]))
    oor = [c for c in S.num if ((win[c] < S.q_lo[c]) | (win[c] > S.q_hi[c])).mean() > 0.2]
    if oor:
        fl.append("many readings outside the baseline range in " + ", ".join(oor[:4]))
    stuck = [c for c in S.num if win[c].notna().sum() >= 20 and win[c].nunique() == 1]
    if stuck:
        fl.append("flat-lined/stuck sensor values in " + ", ".join(stuck[:4]))
    return fl


def explain(source):
    thr, W = S.cfg["thr"], S.cfg["win"]
    hi = sorted([c for c, v in S.psis.items() if v >= thr], key=lambda c: -S.psis[c])
    e = S.err[max(0, S.lp - 100):S.lp]
    up = (np.nanmean(e) - S.ref_err) if len(e) >= 20 else 0.0
    out = []
    if hi and up < 0.05:
        out.append(f"Possible covariate drift: input distribution shifted ({', '.join(hi[:4])}) but error has not clearly risen yet.")
    if hi and up >= 0.05:
        out.append(f"Possible covariate drift hurting accuracy ({', '.join(hi[:4])}; error +{up:.1%}).")
    if (source == "ADWIN" or up >= 0.05) and not hi:
        out.append(f"Possible concept drift: errors changed (+{up:.1%}) while inputs look stable (max PSI {max(S.psis.values(), default=0):.2f}) — the feature→label relationship may have changed.")
    if S.lpsi >= thr / 2:
        out.append(f"Possible target prevalence change: class mix PSI {S.lpsi:.2f} vs baseline labels.")
    fl = quality_flags(S.F.iloc[max(0, S.pos - W):S.pos])
    if fl:
        out.append("Possible sensor-quality change: " + "; ".join(fl) + ".")
    return out or ["Drift signal raised, but no rule-based cause matched; inspect the charts."]


def add_event(source, detail):
    S.events.append(dict(pos=S.pos, time=str(S.T[S.pos - 1]), source=source, detail=detail,
                         explanation=" | ".join(explain(source))))


def monitor():
    pos, W, thr = S.pos, S.cfg["win"], S.cfg["thr"]
    if pos < 30:
        return False
    win = S.F.iloc[max(0, pos - W):pos]
    psis = {}
    for c, spec in S.ref.items():
        p = col_props(spec, win[c])
        if p is not None:
            psis[c] = psi_vec(spec["props"], p)
    lpsi = 0.0
    if S.lp >= 30:
        cnt = np.bincount(S.ys[max(0, S.lp - W):S.lp], minlength=S.cfg["k"]).astype(float)
        lpsi = psi_vec(S.label_ref, cnt / cnt.sum())
    mx = max(psis.values(), default=0.0)
    S.psis, S.lpsi = psis, lpsi
    S.psi_hist.append(dict(t=S.T[pos - 1], max=mx, label=lpsi))
    score = max(mx, lpsi)
    if score >= thr and not S.psi_on:
        S.psi_on = True
        add_event("PSI", f"max feature PSI {mx:.2f}, label PSI {lpsi:.2f} (threshold {thr})")
        return True
    if score < thr * 0.75:
        S.psi_on = False
    return False


def step(batch):
    n, lo = len(S.ys), S.pos
    hi = min(n, lo + batch)
    S.pred[lo:hi] = S.model.predict(S.Xs[lo:hi])
    S.ver[lo:hi] = S.version
    S.pos = hi
    new_lp = n if hi >= n else max(S.lp, hi - S.cfg["delay"])  # labels arrive after a delay
    hit = False
    for i in range(S.lp, new_lp):
        e = float(S.pred[i] != S.ys[i])
        S.err[i] = e
        S.adwin.update(e)
        hit |= bool(S.adwin.drift_detected)
    S.lp = new_lp
    ev = monitor()
    if hit:
        add_event("ADWIN", f"ADWIN flagged a change in the delayed error stream (δ={S.cfg['delta']})")
    return ev or hit


# ───────────────────────── retraining ─────────────────────────
def compat(method):
    info = MODELS[S.model_name]
    if method in (INC, ONL, CON) and not info["partial"]:
        return f"**{S.model_name}** has no `partial_fit`, so '{method}' is unavailable. Use an SGD or MLP baseline, or Full retraining."
    if method == TRA and not info["transfer"]:
        return ("Transfer learning is disabled: tree ensembles have no weights to carry over, and the feature space/classes "
                "are unchanged, so there is nothing to transfer to. Use Full retraining instead.")
    return None


def select_training(strategy, W):
    lp, Xs, ys = S.lp, S.Xs[:S.lp], S.ys[:S.lp]
    start = min(max(0, S.events[-1]["pos"] - S.cfg["win"]), lp) if S.events else 0
    if strategy == STRATS[0]:
        X, y = Xs[start:], ys[start:]
    elif strategy == STRATS[1]:
        X, y = np.vstack([S.Xtr, Xs[start:]]), np.concatenate([S.ytr, ys[start:]])
    else:
        X, y = Xs[-W:], ys[-W:]
    return X, y, start


def train_candidate(strategy, method, W, epochs, bs, replay_n, ratio):
    X, y, start = select_training(strategy, W)
    if len(y) < 30:
        raise ValueError(f"Only {len(y)} labelled rows are available for '{strategy}' (need ≥30). Let the stream run longer.")
    rng, cur, k = np.random.default_rng(SEED), S.model, S.cfg["k"]
    if method == FULL:
        m = make_model(S.model_name)
        m.fit(X, y)
    elif method == TRA:
        if len(np.unique(y)) < k:
            raise ValueError("Transfer learning needs every class in the training data. Use 'Old data + new data' or a larger window.")
        m = copy.deepcopy(cur)
        if isinstance(m, SGDClassifier):
            m.fit(X, y, coef_init=cur.coef_, intercept_init=cur.intercept_)
        else:
            m.set_params(warm_start=True, **({"max_iter": 100} if isinstance(m, MLPClassifier) else {}))
            m.fit(X, y)
    else:
        m = copy.deepcopy(cur)
        if method == INC:
            for _ in range(epochs):
                idx = rng.permutation(len(X))
                for s in range(0, len(idx), bs):
                    m.partial_fit(X[idx[s:s + bs]], y[idx[s:s + bs]])
        elif method == ONL:  # one sample at a time, in arrival order (most recent 1500)
            for i in range(max(0, len(X) - 1500), len(X)):
                m.partial_fit(X[i:i + 1], y[i:i + 1])
        else:  # continual: mix each mini-batch with replayed old samples
            pool_X, pool_y = np.vstack([S.Xtr, S.Xs[:start]]), np.concatenate([S.ytr, S.ys[:start]])
            r = rng.choice(len(pool_X), min(replay_n, len(pool_X)), replace=False)
            bufX, bufy = pool_X[r], pool_y[r]
            for _ in range(epochs):
                idx = rng.permutation(len(X))
                for s in range(0, len(idx), bs):
                    b = idx[s:s + bs]
                    j = rng.choice(len(bufX), max(1, int(len(b) * ratio)))
                    m.partial_fit(np.vstack([X[b], bufX[j]]), np.concatenate([y[b], bufy[j]]))
    ev = lambda mod: dict(hist=metrics(mod, S.Xh, S.yh, k), recent=metrics(mod, S.Xr, S.yr, k))
    return dict(model=m, method=method, strategy=strategy, n=len(y), cur=ev(cur), new=ev(m), label=S.model_label)


# ───────────────────────── plotting ─────────────────────────
def cm_fig(cm, title):
    fig = go.Figure(go.Heatmap(z=cm, x=S.names, y=S.names, text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis_autorange="reversed")
    return fig


def vline(fig, x, color, **kw):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=color, dash="dot", width=1.5), **kw)


def tmap(rows):
    return pd.DataFrame({k: {"Accuracy": v["Accuracy"], "Precision": v["Precision"], "Recall": v["Recall"], "F1": v["F1"]}
                         for k, v in rows.items()}).T.round(4)


# ───────────────────────── UI ─────────────────────────
st.title("📡 IoT Model Drift Detection & Retraining Dashboard")
st.caption("Upload a CSV, train a baseline, replay the data as a stream, watch drift detectors, retrain, and decide whether to accept the candidate.")
if "ready" not in S:
    S.ready = False

# 1 ─ setup
st.header("1 · Data & baseline model")
up = st.file_uploader("Upload a CSV file", type="csv")
if up is not None:
    df = load_csv(up.getvalue())
    cols = list(df.columns)
    st.caption(f"{len(df):,} rows × {len(cols)} columns")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target column (class label)", cols, index=len(cols) - 1)
    ts_opts = ["(none – use row order)"] + [c for c in cols if c != target]
    guess = next((c for c in ts_opts[1:] if datetime_like(df[c]) and not pd.api.types.is_numeric_dtype(df[c])), None)
    ts_sel = c2.selectbox("Timestamp column (optional)", ts_opts, index=ts_opts.index(guess) if guess else 0)
    ts = None if ts_sel.startswith("(none") else ts_sel
    cand_cols = [c for c in cols if c not in (target, ts)]
    default = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c])][:12]
    feats = c3.multiselect("Feature columns", cand_cols, default=default, key=f"feats_{target}_{ts}")
    num, cat, issues = check_features(df, feats, target, ts)
    for m in issues:
        st.warning(m)
    if ts is None:
        st.info("No timestamp selected: rows are used in file order as the chronological order.")
    else:
        st.info("Rows will be sorted by the timestamp column before the chronological split.")

    c1, c2, c3, c4 = st.columns(4)
    model_name = c1.selectbox("Baseline model", list(MODELS))
    base_frac = c2.slider("Baseline fraction (chronological)", 0.2, 0.7, 0.4, 0.05)
    hist_frac = c3.slider("Historical hold-out share of baseline", 0.1, 0.4, 0.25, 0.05)
    delay = c4.number_input("Label delay (rows)", 0, 5000, 50, 10)
    info = MODELS[model_name]
    if not info["partial"]:
        st.warning(f"{model_name} has no `partial_fit`: Incremental, Online and Continual retraining will be disabled for it."
                   + (" Transfer learning stays available (warm-start)." if info["transfer"] else " Transfer learning is also disabled."))
    with st.expander("Drift-detection settings"):
        d1, d2, d3, d4 = st.columns(4)
        bins = d1.slider("PSI reference bins", 5, 20, 10)
        win = d2.slider("PSI window (rows)", 50, 1000, 300, 25)
        thr = d3.slider("PSI alert threshold", 0.05, 0.5, 0.2, 0.05)
        delta = d4.select_slider("ADWIN δ (sensitivity)", [0.0001, 0.001, 0.002, 0.01, 0.05], 0.002)
    sig = (up.name, up.size, target, tuple(num), tuple(cat), ts, model_name, base_frac, hist_frac, delay, bins, win, thr, delta)
    if st.button("🧠 Train baseline & prepare stream", type="primary", disabled=not (num or cat)):
        try:
            with st.spinner("Training baseline…"):
                prepare(df, target, num, cat, ts, model_name, base_frac, hist_frac, int(delay), bins, win, thr, delta)
                S.sig = sig
        except Exception as e:
            S.ready = False
            st.error(str(e))
    if S.ready and S.sig != sig:
        st.warning("Settings changed since the baseline was trained — click the button again to apply them.")
elif not S.ready:
    st.info("Upload a CSV to begin. Nothing is bundled or generated for you.")
    st.stop()

if not S.ready:
    st.stop()

c = S.counts
st.caption(f"Splits (chronological): baseline-train {c['train']:,} · historical hold-out {c['hist']:,} · "
           f"stream (replayed) {c['stream']:,} · recent hold-out {c['recent']:,} (every 5th post-baseline row, never replayed or trained on).")
bm = S.base_metrics
b1, b2, b3, b4 = st.columns(4)
b1.metric("Baseline accuracy (historical hold-out)", f"{bm['hist']['Accuracy']:.3f}")
b2.metric("Baseline F1 (historical hold-out)", f"{bm['hist']['F1']:.3f}")
b3.metric("Baseline accuracy (recent hold-out)", f"{bm['recent']['Accuracy']:.3f}")
b4.metric("Baseline F1 (recent hold-out)", f"{bm['recent']['F1']:.3f}")

# 2 ─ stream
st.header("2 · Stream simulation & live monitoring")
finished = S.pos >= len(S.ys)
k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
if k1.button("▶ Start", disabled=S.started or finished, use_container_width=True):
    S.running = S.started = True
if k2.button("⏸ Pause", disabled=not S.running, use_container_width=True):
    S.running = False
if k3.button("⏯ Resume", disabled=S.running or not S.started or finished, use_container_width=True):
    S.running = True
if k4.button("⟲ Reset", use_container_width=True):
    reset_stream()
batch = k5.slider("Rows per tick", 1, 100, 10)
interval = st.slider("Seconds between ticks", 0.2, 3.0, 0.8, 0.1)


@st.fragment(run_every=interval if S.running else None)
def live_panel():
    if S.running:
        ev = step(batch)
        if S.pos >= len(S.ys) or ev:  # keep streaming after drift; just refresh the rest of the page
            if S.pos >= len(S.ys):
                S.running = False
            st.rerun()
    pos, lp, W = S.pos, S.lp, S.cfg["win"]
    err = S.err[max(0, lp - 100):lp]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Stream position", f"{pos:,}/{len(S.ys):,}")
    m2.metric("Labels received", f"{lp:,}")
    m3.metric("Rolling accuracy (last 100 labelled)", f"{1 - np.nanmean(err):.3f}" if len(err) else "–")
    m4.metric("Max feature PSI", f"{max(S.psis.values(), default=0):.3f}")
    m5.metric("Anomaly rate", f"{S.anom[:pos].mean():.1%}" if pos else "–")
    m6.metric("Drift events", len(S.events))
    status = "▶ running" if S.running else ("✔ finished" if finished else "⏸ paused/idle")
    if S.events and pos - S.events[-1]["pos"] <= 3 * W:
        e = S.events[-1]
        st.error(f"🚨 **Drift alert ({e['source']})** at row {e['pos']}: {e['detail']}\n\n" + "\n".join(f"- {x}" for x in e["explanation"].split(" | "))
                 + f"\n\n*Stream is still running ({status}). Retrain in section 3.*")
    else:
        st.success(f"No active drift alert · stream {status}")
    if pos == 0:
        st.info("Press **Start** to replay the stream.")
        return
    lo = max(0, pos - 300)
    x = S.T[lo:pos]
    sel = st.multiselect("Sensors to plot (numeric features)", S.num, default=S.num[:2], max_selections=3, key="plot_sel")
    g1, g2 = st.columns(2)
    if sel:
        fig = make_subplots(rows=len(sel), cols=1, shared_xaxes=True, vertical_spacing=0.05)
        an = np.where(S.anom[lo:pos])[0]
        for i, col in enumerate(sel, 1):
            v = S.F[col].to_numpy()[lo:pos]
            fig.add_trace(go.Scatter(x=x, y=v, mode="lines", name=col), row=i, col=1)
            fig.add_trace(go.Scatter(x=[x[j] for j in an], y=v[an], mode="markers", marker=dict(color="red", size=7),
                                     name="anomaly (|z|>4)", showlegend=i == 1), row=i, col=1)
        for e in S.events:
            if lo < e["pos"] <= pos:
                vline(fig, S.T[e["pos"] - 1], "crimson")
        fig.update_layout(title="Incoming sensor values & anomalies", height=380, margin=dict(l=10, r=10, t=40, b=10))
        g1.plotly_chart(fig, use_container_width=True)
    else:
        g1.info("No numeric features selected/available for the sensor chart.")
    pr, er = S.pred[lo:pos], S.err[lo:pos]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
    for lab, colr, mask in (("correct", "green", er == 0), ("wrong", "red", er == 1), ("label pending", "gray", np.isnan(er))):
        j = np.where(mask)[0]
        fig.add_trace(go.Scatter(x=[x[i] for i in j], y=[S.names[pr[i]] for i in j], mode="markers", name=lab,
                                 marker=dict(color=colr, size=6)), row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=S.names, row=1, col=1)
    if lp > 10:
        a = max(0, lp - 300)
        roll = pd.Series(S.err[:lp]).rolling(50, min_periods=10).mean().iloc[a:lp]
        fig.add_trace(go.Scatter(x=S.T[a:lp], y=roll, mode="lines", name="rolling error (50)", line=dict(color="orange")), row=2, col=1)
    for e in S.events:
        if e["source"] == "ADWIN" and lo < e["pos"] <= pos:
            vline(fig, S.T[e["pos"] - 1], "purple")
    fig.update_layout(title="Model predictions & delayed error", height=380, margin=dict(l=10, r=10, t=40, b=10))
    g2.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    if S.psi_hist:
        h = pd.DataFrame(S.psi_hist).tail(600)
        fig = go.Figure([go.Scatter(x=h.t, y=h["max"], name="max feature PSI"), go.Scatter(x=h.t, y=h.label, name="label PSI")])
        fig.add_hline(y=S.cfg["thr"], line_dash="dash", line_color="red")
        for src, sym in (("PSI", "star"), ("ADWIN", "diamond")):
            ev = [e for e in S.events if e["source"] == src]
            fig.add_trace(go.Scatter(x=[S.T[e["pos"] - 1] for e in ev], y=[S.cfg["thr"]] * len(ev), mode="markers", name=f"{src} alert",
                                     marker=dict(symbol=sym, size=13, color="crimson" if src == "PSI" else "purple")))
        fig.update_layout(title="Drift history (PSI over time, alerts marked)", height=330, margin=dict(l=10, r=10, t=40, b=10))
        g3.plotly_chart(fig, use_container_width=True)
    if S.psis:
        p = pd.Series(S.psis).sort_values().tail(15)
        fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h", marker_color=["crimson" if v >= S.cfg["thr"] else "steelblue" for v in p.values]))
        fig.update_layout(title="Current PSI per feature", height=330, margin=dict(l=10, r=10, t=40, b=10))
        g4.plotly_chart(fig, use_container_width=True)
    if S.events:
        st.subheader("Drift event log")
        st.dataframe(pd.DataFrame(S.events)[["pos", "time", "source", "detail", "explanation"]].iloc[::-1], use_container_width=True, hide_index=True)


live_panel()

# 3 ─ retraining
st.header("3 · Retraining")
r1, r2 = st.columns(2)
strategy = r1.radio("Training-data strategy", STRATS, horizontal=True)
method = r2.selectbox("Retraining method", METHODS)
p1, p2, p3, p4 = st.columns(4)
W = p1.number_input("Sliding-window size (labelled rows)", 50, 20000, 500, 50)
epochs = p2.number_input("Epochs (incremental / continual)", 1, 20, 3)
replay_n = p3.number_input("Replay buffer size", 50, 20000, 500, 50)
ratio = p4.slider("Replay ratio per mini-batch", 0.2, 2.0, 1.0, 0.1)
st.caption(f"Labelled stream rows available: **{S.lp:,}** · 'New data' starts one PSI-window before the latest drift alert "
           "(or at stream start if none). Online learning updates one sample at a time (latest ≤1500). "
           "Preprocessing is kept fixed so every model shares one feature space.")
msg = compat(method)
if msg:
    st.warning(msg)
if st.button("🔁 Train candidate model", disabled=bool(msg) or S.lp < 30):
    try:
        with st.spinner("Training candidate…"):
            S.cand = train_candidate(strategy, method, int(W), int(epochs), 64, int(replay_n), ratio)
    except Exception as e:
        st.error(f"Retraining failed: {e}")
if S.lp < 30:
    st.info("Run the stream until at least 30 labels have arrived to enable retraining.")

# 4 ─ comparison & decision
st.header("4 · Before / after comparison & model decision")
cd = S.cand
if cd is None:
    st.info("Train a candidate to compare it against the current model on the fixed held-out sets.")
else:
    st.caption(f"Candidate: **{cd['method']}** · **{cd['strategy']}** · trained on {cd['n']:,} rows. "
               "Evaluated on fixed held-out data that no model was trained on.")
    t1, t2 = st.columns(2)
    for col, key, title in ((t1, "recent", "Recent hold-out (post-baseline period)"), (t2, "hist", "Retained historical hold-out")):
        tbl = tmap({"Current": cd["cur"][key], "Candidate": cd["new"][key]})
        tbl.loc["Δ (cand − cur)"] = (tbl.loc["Candidate"] - tbl.loc["Current"]).round(4)
        col.subheader(title)
        col.dataframe(tbl, use_container_width=True)
    forget = cd["new"]["hist"]["Accuracy"] - cd["cur"]["hist"]["Accuracy"]
    st.caption(f"Performance on retained historical data changed by {forget:+.3f} accuracy "
               + ("(possible forgetting)." if forget < -0.02 else "(no notable forgetting)."))
    q = st.columns(4)
    q[0].plotly_chart(cm_fig(cd["cur"]["recent"]["cm"], "Current · recent"), use_container_width=True)
    q[1].plotly_chart(cm_fig(cd["new"]["recent"]["cm"], "Candidate · recent"), use_container_width=True)
    q[2].plotly_chart(cm_fig(cd["cur"]["hist"]["cm"], "Current · historical"), use_container_width=True)
    q[3].plotly_chart(cm_fig(cd["new"]["hist"]["cm"], "Candidate · historical"), use_container_width=True)
    a1, a2, _ = st.columns([1, 1, 3])
    if a1.button("✅ Accept candidate", type="primary"):
        S.version += 1
        S.model_label = f"v{S.version} · {cd['method']} · {cd['strategy']}"
        S.model = cd["model"]
        S.adwin = ADWIN(delta=S.cfg["delta"])  # restart error monitoring for the new model
        S.decisions.append(dict(decision="accepted", model=S.model_label, recent_F1=round(cd["new"]["recent"]["F1"], 4)))
        S.cand = None
        st.rerun()
    if a2.button("❌ Reject candidate"):
        S.decisions.append(dict(decision="rejected", model=f"{cd['method']} · {cd['strategy']}", recent_F1=round(cd["new"]["recent"]["F1"], 4)))
        S.cand = None
        st.rerun()

st.subheader("Current model status")
s1, s2, s3 = st.columns(3)
s1.metric("Serving model", S.model_label)
s2.metric("Accepted retrains", sum(d["decision"] == "accepted" for d in S.decisions))
s3.metric("Rejected candidates", sum(d["decision"] == "rejected" for d in S.decisions))
st.caption(f"Base estimator: {S.model_name}. Rejected candidates are discarded; the previous model keeps serving the stream.")
if S.decisions:
    st.dataframe(pd.DataFrame(S.decisions), use_container_width=True, hide_index=True)
