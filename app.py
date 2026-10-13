"""Model Drift Dashboard: Streamlit UI.

All modelling, drift detection and decision logic lives in `drift_core.py`; this file only renders state and
forwards user actions to a `drift_core.Session`.

Run with:  streamlit run app.py      (requires: pip install streamlit plotly river scikit-learn pandas)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import drift_core as core
from drift_core import KEEP, REG, STRATEGIES, Config, Session

CSV_PATH = Path(__file__).parent / "synthetic_iot_sensor_data.csv"   # keep it next to app.py
PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]
CSS = """<style>
.block-container{padding-top:1.4rem;padding-bottom:1.5rem}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border-radius:8px;padding:.45rem .8rem}
</style>"""

_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner="Reading CSV...")
def read(path: str) -> pd.DataFrame:
    return core.read_csv_bytes(Path(path).read_bytes())


def wide(fn, *a, **kw):
    kw["width" if _NEW_API else "use_container_width"] = "stretch" if _NEW_API else True
    return fn(*a, **kw)


def flash(level, text):
    st.session_state["flash"] = (level, text)


def confirm_box(flag, message, yes_label):
    if not st.session_state.get(flag):
        return False
    st.warning(message)
    a, b = st.columns(2)
    if a.button(yes_label, key=flag + "_yes", type="primary"):
        st.session_state[flag] = False
        return True
    if b.button("Cancel", key=flag + "_no"):
        st.session_state[flag] = False
        st.rerun()
    return False


# ------------------------------------------------------------------ charts
def drift_chart(S: Session):
    n = min(S.k, len(S.ver), len(S.errs), len(S.thr))
    xs, ver = np.arange(1, n + 1), np.asarray(S.ver)[:n]
    errs, thr = np.asarray(S.errs, float)[:n], np.asarray(S.thr, float)[:n]
    fig = go.Figure()
    for i, v in enumerate(sorted(np.unique(ver))):
        idx = np.flatnonzero(ver == v)
        fig.add_trace(go.Scatter(x=xs[idx], y=errs[idx], mode="lines+markers", name=f"Model v{v}",
                                 line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.add_trace(go.Scatter(x=xs, y=thr, mode="lines", name="Drift threshold",
                             line=dict(dash="dash", color="#c62828")))
    for r in S.retrains:
        if 1 <= r.applied + 1 <= n:
            fig.add_vline(x=r.applied + 1, line_color="#2e7d32", annotation_text=f"Retrain v{r.version}",
                          annotation_position="top left")
        if 1 <= r.start + 1 <= n:
            fig.add_vline(x=r.start + 1, line_color="#c62828", line_dash="dot", opacity=0.5)
    if S.alert is not None and 1 <= S.alert.start + 1 <= n:
        fig.add_vline(x=S.alert.start + 1, line_color="#c62828", annotation_text="Drift starts",
                      annotation_position="top right")
    fig.update_layout(title="Prediction error per incoming batch", xaxis_title="Batch",
                      yaxis_title="MAE (target units)" if S.task == "regression"
                      else "Error rate (share of wrong predictions)",
                      height=360, margin=dict(t=50, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.2))
    wide(st.plotly_chart, fig)


def confusion_fig(y, p):
    labels, cm = core.confusion(y, p)
    fig = go.Figure(go.Heatmap(z=cm, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels], text=cm,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
    return fig


# ------------------------------------------------------------------ panels
def decision_panel(S: Session):
    c, al = S.cand, S.alert
    with st.container(border=True):
        st.subheader("Drift confirmed: choose how to adapt")
        if c.get("error"):
            st.error(c["error"])
            if st.button("Skip retraining (keep current model)", key="skip_err"):
                S.keep("not enough data to evaluate candidates")
                flash("info", "Kept the current model.")
                st.rerun()
            return
        m = st.columns(4)
        m[0].metric("Drift started at batch", al.start + 1)
        m[1].metric("Historical labelled rows", f"{c['n_old']:,}")
        m[2].metric("New labelled rows", f"{c['n_new']:,}", help=f"{c['n_train']:,} used for training candidates, "
                    f"{c['hold']:,} held out")
        m[3].metric("Held-out evaluation rows", f"{c['hold']:,}")
        t = c["table"]
        ts = t.sort_values("Selection score").reset_index(drop=True)
        cols = ["#9e9e9e" if s == KEEP else "#2e7d32" if s == c["rec"] else "#1f6fb2" for s in ts["Strategy"]]
        fig = go.Figure(go.Bar(x=ts["Selection score"], y=ts["Strategy"], orientation="h", marker_color=cols,
                               text=ts["Selection score"].round(4), textposition="auto"))
        fig.update_layout(title=f"Held-out {c['metric']} per candidate (lower is better)", height=60 + 38 * len(ts),
                          yaxis=dict(autorange="reversed"), margin=dict(t=45, b=30, l=10, r=10),
                          xaxis_title=c["metric"])
        wide(st.plotly_chart, fig)
        st.markdown(c["why"])
        for w in c["warnings"]:
            st.warning(w)
        score = dict(zip(t["Strategy"], t["Selection score"]))
        opts = [s.name for s in STRATEGIES if s.name in score and np.isfinite(score[s.name])]
        k = S.k
        choice = st.radio("Strategy (held-out score in brackets)", opts, index=opts.index(c["rec"]),
                          format_func=lambda n: f"{n}  ({score[n]:.4f})", key=f"pick_{k}")
        st.caption(f"{REG[choice].blurb}  \nHow it will be built: {c['notes'][choice]}.")
        a1, a2, _ = st.columns([1.2, 1.6, 3])
        if a1.button("Apply", type="primary", disabled=choice == KEEP, key=f"apply_{k}",
                     help="Keep Current Model is confirmed with the Skip button." if choice == KEEP else None):
            ok, msg = S.apply(choice)
            flash("success" if ok else "error", msg)
            st.rerun()
        if a2.button("Skip retraining (keep current model)", key=f"skip_{k}"):
            flash("info", S.keep())
            st.rerun()
        if c["failed"]:
            st.caption("Failed to build, excluded from the ranking: " + "; ".join(f"{n} ({why})" for n, why in c["failed"]))
        with st.expander("Detailed held-out metrics and confusion matrix"):
            wide(st.dataframe, t.round(4), hide_index=True)
            if S.task == "classification":
                st.caption(f"Fault class = '{S.pos}'. Precision/recall/F1 are for that class.")
                st.write(f"Confusion matrix: {choice}")
                wide(st.plotly_chart, confusion_fig(c["yh"], c["preds"][choice]))


def retrain_results(S: Session):
    st.subheader("How is each retrained model performing now?")
    wide(st.dataframe, S.retrain_summary().round(4), hide_index=True)
    st.caption("Held-out columns use the decision's selection metric; live columns use the monitoring metric "
               f"({S.unit}). Early live numbers cover few batches, so read them with care.")


# ------------------------------------------------------------------ sidebar
def sidebar_config(S):
    with st.sidebar:
        st.header("Configuration")
        if not CSV_PATH.exists():
            st.error(f"Data file not found: `{CSV_PATH.name}`. Put it in the same folder as app.py.")
            return
        try:
            df = read(str(CSV_PATH))
        except Exception as exc:
            st.error(str(exc))
            return
        st.caption(f"Data: `{CSV_PATH.name}` ({len(df):,} rows). Rows should be in time order, or pick a time column.")
        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        tcol = st.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
        st.caption("Model: Hoeffding Adaptive Tree (river), learns one row at a time and adapts to drift internally.")
        tasks = ["classification", "regression"]
        task = st.radio("Task", tasks, index=tasks.index(core.guess_task(df, target)), horizontal=True)
        train_pct = st.slider("Initial training %", 20, 70, 40, 5)
        B = int(st.number_input("Batch size (rows)", 20, 5000, 100))
        sens = st.selectbox("Drift sensitivity", list(core.SENS), index=1)
        delay = int(st.number_input("Label delay (batches)", 0, 20, 0))
        min_rows = int(st.number_input("Min new labelled rows before asking", 100, 100000, 300,
                                       help="After a drift alert, monitoring continues until this many post-drift "
                                            "labelled rows exist."))
        snooze = int(st.number_input("Snooze after 'keep' (batches)", 0, 200, 10))
        with st.expander("Advanced settings"):
            hold_frac = st.slider("Share of new labelled rows held out for evaluation", 0.10, 0.40, 0.25, 0.05)
            metric = st.selectbox("Selection metric (lower is better)", core.METRICS[task])
            pos = None
            if task == "classification":
                labs, dflt = core.class_options(df[target])
                if labs:
                    pos = st.selectbox("Fault (positive) class", labs, index=labs.index(dflt))
            st.markdown("**Hoeffding Adaptive Tree settings**")
            grace = int(st.number_input("Grace period (rows between split attempts)", 10, 5000, 200, step=10,
                                        help="Lower = the tree splits sooner and adapts faster, but is noisier."))
            delta = st.select_slider("Split confidence (delta, smaller = more cautious)", core.DELTAS, value=1e-7,
                                     format_func=lambda v: f"{v:.0e}")
            tau = st.number_input("Tie threshold (tau)", 0.001, 0.5, 0.05, step=0.01, format="%.3f")
            adwin = st.select_slider("ADWIN sensitivity (larger = reacts to drift sooner)", core.ADWIN_DELTAS,
                                     value=0.002, help="Confidence of the drift detector inside each tree node.")
            win_thr = int(st.number_input("Alternate-tree window (rows)", 50, 5000, 300, step=50,
                                          help="Rows an alternate subtree must see before it can replace "
                                               "the original branch."))
            seed = int(st.number_input("Random seed", 0, 10_000, 42))
        running = S is not None and S.k > 0
        go_ = False
        if st.button("Restart with these settings" if S is not None else "Start monitoring", type="primary"):
            if running:
                st.session_state["confirm_restart"] = True
            else:
                go_ = True
        if confirm_box("confirm_restart", "This discards the current session (model versions, events, history).",
                       "Yes, restart"):
            go_ = True
        if go_:
            cfg = Config(train_pct=train_pct, B=B, z=core.SENS[sens], delay=delay, min_rows=min_rows, snooze=snooze,
                         hold_frac=hold_frac, metric=metric, pos=pos, seed=seed, grace=grace, delta=delta,
                         tau=float(tau), adwin=adwin, win_thr=win_thr)
            try:
                with st.spinner("Training the first Hoeffding Adaptive Tree..."):
                    st.session_state["S"] = Session.create(df, target, tcol, task, cfg)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.pop("flash", None)
            st.rerun()


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 Model Drift Dashboard")
    if not core.RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install river` and restart.")
        st.stop()
    sidebar_config(st.session_state.get("S"))
    S = st.session_state.get("S")
    if S is None:
        st.info("Choose your columns in the sidebar and click **Start monitoring**.")
        st.stop()

    off = not S.can_advance
    with st.container(border=True):
        b0, b1, b2, b3 = st.columns([2.4, 1, 1, 1])
        if b0.button("▶ Run until a decision is needed", type="primary", disabled=off, key="run_all"):
            S.run(None)
        if b1.button("Next batch", disabled=off, key="run_1"):
            S.run(1)
        if b2.button("Next 10 batches", disabled=off, key="run_10"):
            S.run(10)
        if b3.button("Reset", key="reset"):
            if S.k > 0:
                st.session_state["confirm_reset"] = True
            else:
                st.session_state.pop("S")
                st.rerun()
    if confirm_box("confirm_reset", "Reset the session? Model versions, events and history will be discarded.",
                   "Yes, reset"):
        st.session_state.pop("S")
        st.session_state.pop("flash", None)
        st.rerun()

    m = st.columns(4)
    m[0].metric("Batches received", f"{S.k} / {S.nb}")
    m[1].metric("Deployed model", f"Adaptive tree v{S.version}")
    m[2].metric("Retraining events", len(S.retrains))
    m[3].metric("Status", S.status)

    if st.session_state.get("flash"):
        lvl, txt = st.session_state.pop("flash")
        getattr(st, lvl)(txt)

    if S.k:
        drift_chart(S)

    if S.alert and S.alert.status == "waiting":
        st.info(f"Drift suspected since batch {S.alert.start + 1}, but only {S.new_rows:,} of {S.cfg.min_rows:,} new "
                "labelled rows have arrived. Not enough to retrain safely, so monitoring continues. You will be asked "
                "once there is enough data, or the alert clears itself if the error returns to normal.")

    if S.ready:
        with st.spinner("Testing incremental learning on the newest labelled rows..."):
            S.ensure_candidates()
        decision_panel(S)

    if S.retrains:
        retrain_results(S)
    if S.events:
        with st.expander("Event log", expanded=False):
            wide(st.dataframe, pd.DataFrame(S.events), hide_index=True)


if __name__ == "__main__":
    main()
