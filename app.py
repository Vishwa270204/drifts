"""Model Drift Dashboard (Streamlit). Run with:  streamlit run drift_app.py

One Hoeffding Adaptive Tree learns from every labelled batch automatically (see drift_core.py). The dashboard
ADWIN only watches the prediction errors and shows informational alerts; it never pauses, resets or replaces the model.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import drift_core as dc

st.set_page_config(page_title="Model Drift Dashboard", layout="wide")
st.title("Model Drift Dashboard")
st.caption("A Hoeffding Adaptive Tree predicts each incoming batch, then learns from its labels automatically. "
           "A separate ADWIN monitor watches the prediction errors and raises alerts. Alerts are informational.")

if not dc.RIVER_OK:
    st.error("The `river` package is not installed. Run: pip install river")
    st.stop()

ss = st.session_state
for key, val in dict(sess=None, running=False, shown=0, sig=None).items():
    ss.setdefault(key, val)
if ss.sess is not None and getattr(ss.sess, "SCHEMA", None) != dc.SCHEMA:
    ss.sess, ss.running, ss.shown = None, False, 0          # created by older code

# ---------------------------------------------------------------- data
up = st.file_uploader("Upload a CSV file with a labelled target column", type=["csv"])
if up is None:
    st.info("Upload a CSV to begin. The file needs a target column with true labels (classes or numbers).")
    st.stop()
sig = (up.name, up.size)
if ss.sig != sig:
    ss.sig, ss.sess, ss.running, ss.shown = sig, None, False, 0
try:
    df = dc.read_csv_bytes(up.getvalue())
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if df.shape[1] < 2 or len(df) == 0:
    st.error("The file needs at least one feature column, one target column and some rows.")
    st.stop()

# ---------------------------------------------------------------- controls
locked = ss.sess is not None
c = st.columns(4)
target = c[0].selectbox("Target column", list(df.columns), index=len(df.columns) - 1, disabled=locked)
others = [x for x in df.columns if x != target]
guess = dc.detect_time_col(df, target)
tcol = c[1].selectbox("Time order", ["(row order)"] + others,
                      index=(others.index(guess) + 1) if guess in others else 0, disabled=locked)
task = c[2].selectbox("Task", ["classification", "regression"],
                      index=0 if dc.guess_task(df, target) == "classification" else 1, disabled=locked)
train_pct = c[3].slider("Initial training (%)", 20, 80, 40, disabled=locked)
c = st.columns(4)
batch = c[0].number_input("Batch size (rows)", 10, 5000, 100, step=10, disabled=locked)
sens = c[1].selectbox("Alert sensitivity", list(dc.SENS), index=1, disabled=locked)
delay = c[2].number_input("Label delay (batches)", 0, 100, 0, disabled=locked)
c[3].slider("Speed (steps per refresh)", 1, 20, 2, key="speed")

b1, b2, _ = st.columns([1, 1, 6])
message = None
label = "Start" if ss.sess is None else ("Pause" if ss.running else "Resume")
if b1.button(label, type="primary", disabled=bool(ss.sess is not None and ss.sess.done)):
    if ss.sess is None:
        try:
            cfg = dc.Config.simple(int(batch), sens)
            cfg.train_pct, cfg.delay = int(train_pct), int(delay)
            ss.sess = dc.Session.create(df, target, tcol, task, cfg)
            ss.running, ss.shown = True, 0
            st.rerun()
        except ValueError as exc:
            message = str(exc)
    else:
        ss.running = not ss.running
        st.rerun()
if b2.button("Reset"):
    ss.sess, ss.running, ss.shown = None, False, 0
    st.rerun()
if message:
    st.error(message)


# ---------------------------------------------------------------- live view
def figure(s):
    fig = go.Figure()
    x = list(range(1, len(s.errs) + 1))
    fig.add_scatter(x=x, y=s.errs, mode="lines", name=f"Batch {s.unit}", line=dict(width=2))
    if s.levels:
        lv = sorted(s.levels.items())
        fig.add_scatter(x=[j + 1 for j, _ in lv], y=[v for _, v in lv], mode="lines", name="ADWIN reference level",
                        line=dict(dash="dash", width=2))
    if s.alerts:
        fig.add_scatter(x=[a.batch + 1 for a in s.alerts], y=[s.errs[a.batch] for a in s.alerts], mode="markers",
                        name="Error increase detected", marker=dict(symbol="x", size=13, color="crimson"),
                        text=[f"Alert {a.id}" for a in s.alerts], hovertemplate="%{text}: batch %{x}<extra></extra>")
    if s.adaptations:
        fig.add_scatter(x=[a["batch"] + 1 for a in s.adaptations], y=[s.errs[a["batch"]] for a in s.adaptations],
                        mode="markers", name="Tree structure changed (HAT counters)",
                        marker=dict(symbol="diamond-open", size=11))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Batch (labels released)",
                      yaxis_title=s.unit, uirevision="keep", legend=dict(orientation="h", y=1.12))
    if s.task == "classification":
        fig.update_yaxes(tickformat=".0%")
    return fig


def live():
    s = ss.sess
    if s is None:
        st.info("Press Start to replay the file as a live stream.")
        return
    if ss.running and not s.done:
        try:
            s.run(int(st.session_state.get("speed", 2)))
        except dc.DetectorError as exc:
            ss.running = False
            st.error(str(exc))
    for a in s.new_alerts(ss.shown):                           # each alert is announced once
        st.toast(f"Alert {a.id}: error increase detected at batch {a.batch + 1}", icon="⚠️")
    ss.shown = len(s.alerts)

    st.progress(min(s.k / s.nb, 1.0), text=f"{s.rows_seen:,} of {s.N:,} stream rows arrived · "
                                          f"{s.labels_pending} batch(es) waiting for delayed labels")
    (st.warning if s.status.startswith("Error") else st.success)(f"{s.status}  ·  automatic learning is ON "
                                                                  f"({s.learned:,} streamed rows learned)")
    m = st.columns(4)
    m[0].metric(f"Latest {s.metric}", dc.fmt_score(s.metric, s.scores[-1]) if s.scores else "–")
    m[1].metric(f"Mean {s.unit}", s._fmt(sum(s.errs) / len(s.errs)) if s.errs else "–")
    m[2].metric("Error-increase alerts", len(s.alerts))
    counters = s.model.adaptation_counters()
    m[3].metric("Tree structure changes", len(s.adaptations) if counters else "not reported")
    if s.errs:
        st.plotly_chart(figure(s), width="stretch", key="err_chart")
    if s.alerts:
        st.caption("An alert means ADWIN found a statistically significant rise in the error. It does not prove "
                   "concept drift, and the tree is not reset or paused; it keeps learning.")
        st.dataframe(pd.DataFrame([{"Alert": a.id, "Batch": a.batch + 1, "Estimated start": a.start + 1,
                                    "Rows arrived": a.rows_seen, "Error before": s._fmt(a.reference),
                                    "Error in batch": s._fmt(a.recent)} for a in s.alerts]),
                     hide_index=True, width="stretch")
    if s.events:
        with st.expander("Event log"):
            st.dataframe(pd.DataFrame(s.events), hide_index=True, width="stretch")
    if s.done and ss.running:
        ss.running = False
        st.rerun()


active = ss.sess is not None and ss.running and not ss.sess.done
st.fragment(live, run_every=0.5 if active else None)()
