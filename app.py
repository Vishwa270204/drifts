from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import drift_core as core
from drift_core import KEEP, SENS, UPDATE, Config, Session, fmt_score

CSV_PATH = Path(__file__).resolve().parent / "sensor.csv"   # keep it next to app.py
PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]
CSS = """<style>
.block-container{padding-top:2.6rem;padding-bottom:.8rem;max-width:1400px}
[data-testid="stMetric"]{background:rgba(128,128,128,.07);border-radius:10px;padding:.35rem .8rem}
[data-testid="stMetricValue"]{font-size:1.7rem}
[data-testid="stVerticalBlock"]{gap:.6rem}
h3{padding:0!important;margin:0!important}
</style>"""
_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


# ------------------------------------------------------------------ helpers
def wide(fn, *a, **kw):
    kw["width" if _NEW_API else "use_container_width"] = "stretch" if _NEW_API else True
    return fn(*a, **kw)


def flash(level, text):
    st.session_state["flash"] = (level, text)


def confirm_box(flag, message, yes_label):
    """Inline confirmation; True only on the run where the user confirms."""
    if not st.session_state.get(flag):
        return False
    st.warning(message)
    a, b, _ = st.columns([1, 1, 4])
    if a.button(yes_label, key=flag + "_yes", type="primary"):
        st.session_state[flag] = False
        return True
    if b.button("Cancel", key=flag + "_no"):
        st.session_state[flag] = False
        st.rerun()
    return False


@st.cache_data(show_spinner="Reading CSV...")
def read_bytes(data: bytes):
    return core.read_csv_bytes(data)


def load_data():
    """Default CSV if present, otherwise an uploader. Returns (df, name) or (None, None)."""
    if CSV_PATH.exists():
        return read_bytes(CSV_PATH.read_bytes()), CSV_PATH.name
    up = st.file_uploader("Upload a CSV (rows in time order)", type="csv")
    return (read_bytes(up.getvalue()), up.name) if up else (None, None)


# ------------------------------------------------------------------ settings: target, sensitivity, batch size
def settings_panel(S):
    with st.expander("Settings", expanded=S is None):
        try:
            df, name = load_data()
        except Exception as exc:
            st.error(str(exc))
            return
        if df is None:
            return
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        sens = c2.selectbox("Drift sensitivity", list(SENS), index=1)
        batch = int(c3.number_input("Batch size (rows)", 20, 5000, 100, step=10))
        task, tcol = core.guess_task(df, target), core.detect_time_col(df, target)
        st.caption(f"`{name}`: {len(df):,} rows · detected **{task}** · ordered by "
                   f"{'`' + tcol + '`' if tcol else 'row order'}")
        running = S is not None and S.k > 0
        go_ = False
        if st.button("Restart" if S is not None else "Start monitoring", type="primary"):
            if running:
                st.session_state["confirm_restart"] = True
            else:
                go_ = True
        if confirm_box("confirm_restart", "This discards the current session.", "Yes, restart"):
            go_ = True
        if go_:
            try:
                with st.spinner("Training the first model..."):
                    st.session_state["S"] = Session.create(df, target, tcol, task, Config.simple(batch, sens))
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.pop("flash", None)
            st.rerun()


# ------------------------------------------------------------------ run controls, status, help
def controls(S):
    off = not S.can_advance
    a, b, c, d = st.columns([3, 1.2, 1.4, 1])
    if a.button("▶ Run until a decision is needed", type="primary", disabled=off, key="run_all", width="stretch"):
        S.run(None)
    if b.button("Next batch", disabled=off, key="run_1", width="stretch"):
        S.run(1)
    if c.button("Next 10 batches", disabled=off, key="run_10", width="stretch"):
        S.run(10)
    if d.button("Reset", key="reset", width="stretch"):
        if S.k > 0:
            st.session_state["confirm_reset"] = True
        else:
            st.session_state.pop("S")
            st.rerun()
    if confirm_box("confirm_reset", "Reset the session? Model versions and history will be discarded.", "Yes, reset"):
        st.session_state.pop("S")
        st.session_state.pop("flash", None)
        st.rerun()


def status_line(S):
    st.progress(S.k / S.nb if S.nb else 0.0,
                text=f"Batch {S.k} of {S.nb}   ·   Model v{S.version}   ·   {S.status}")


def how_it_works(S):
    z = S.cfg.z if S else 3.0
    B = S.cfg.B if S else 100
    mn = S.cfg.min_rows if S else 400
    with st.popover("ℹ️ How it works"):
        st.markdown(f"""
**How drift is detected**
1. The first 40% of rows train the model; its error on the last fifth of that part sets the *normal* error level.
2. Remaining rows arrive in batches of {B}. Each batch is scored by the deployed model.
3. A batch is **bad** if its error is above `normal + {z:g} × spread / √rows` (the dashed red line).
4. **Two bad batches in a row** raise a drift alert. The app then keeps collecting labelled rows until
   {mn:,} new ones exist. If error returns to normal for 3 batches, the alert clears itself.

**How the recommendation is made**
1. The newest 25% of the new labelled rows are **held out**.
2. A copy of the current model keeps learning from the rest of the new rows.
3. Current and updated models are both scored on the held-out rows. Lower error wins; a tie goes to *keep*.
4. A warning appears if the test sample is small, one class dominates, or the gain is within noise.

If you update, the model learns from **all** new rows and the alert threshold is reset from its held-out errors.
""")


# ------------------------------------------------------------------ chart
def drift_chart(S, height=330):
    n = min(S.k, len(S.ver), len(S.errs), len(S.thr))
    xs, ver = np.arange(1, n + 1), np.asarray(S.ver)[:n]
    errs, thr = np.asarray(S.errs, float)[:n], np.asarray(S.thr, float)[:n]
    pct = S.task == "classification"
    fig = go.Figure()
    for i, v in enumerate(sorted(np.unique(ver))):
        idx = np.flatnonzero(ver == v)
        fig.add_trace(go.Scatter(x=xs[idx], y=errs[idx], mode="lines+markers", name=f"Model v{v}",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=xs, y=thr, mode="lines", name="Alert threshold",
                             line=dict(dash="dash", color="#c62828", width=1.5)))
    for r in S.retrains:
        if 1 <= r.applied + 1 <= n:
            fig.add_vline(x=r.applied + 1, line_color="#2e7d32", line_width=1.5,
                          annotation_text=f"v{r.version}", annotation_position="top left")
    if S.alert is not None and 1 <= S.alert.start + 1 <= n:
        fig.add_vrect(x0=S.alert.start + 1, x1=n, fillcolor="#c62828", opacity=0.07, line_width=0,
                      annotation_text="Drift", annotation_position="top left")
    fig.update_layout(height=height, margin=dict(t=10, b=30, l=50, r=10), hovermode="x unified",
                      xaxis_title="Batch", yaxis_title="Error rate" if pct else "MAE",
                      legend=dict(orientation="h", y=-0.22))
    if pct:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    wide(st.plotly_chart, fig)


# ------------------------------------------------------------------ decision
def decision_panel(S):
    d = S.cand
    with st.container(border=True):
        st.markdown("### ⚠️ Drift detected")
        if d.error:
            st.error(d.error)
            if st.button("Keep current model", key="skip_err"):
                flash("info", S.keep("not enough data to compare"))
                st.rerun()
            return
        st.write(f"Error was above the alert threshold for 2 batches in a row, from **batch {S.alert.start + 1}**.")
        can_update = UPDATE in d.scores and np.isfinite(d.scores[UPDATE])
        cur = fmt_score(d.metric, d.scores[KEEP])
        a, b = st.columns(2)
        a.metric("Current model", cur)
        new = fmt_score(d.metric, d.scores[UPDATE]) if can_update else "n/a"
        if can_update:
            ch = d.change_pct(UPDATE)
            b.metric("Updated model", new, delta=None if np.isnan(ch) else f"{ch:+.1f}%", delta_color="inverse")
        else:
            b.metric("Updated model", new)
        st.caption(f"{d.metric}, measured on the {d.hold} newest labelled rows (lower is better).")
        rec_update = d.rec == UPDATE
        if rec_update:
            st.success(f"**Update the model.** It scores lower on recent data ({new} vs {cur}).")
        else:
            st.info("**Keep the current model.** The update is not better on recent data.")
        if d.notes:
            st.caption("⚠️ Take with care: " + "; ".join(d.notes) + ".")
        c1, c2 = st.columns(2)
        if c1.button("Update model", type="primary" if rec_update else "secondary", disabled=not can_update,
                     key=f"apply_{S.k}", width="stretch"):
            ok, msg = S.apply(UPDATE)
            flash("success" if ok else "error", msg)
            st.rerun()
        if c2.button("Keep current model", type="secondary" if rec_update else "primary", key=f"keep_{S.k}",
                     width="stretch"):
            flash("info", S.keep())
            st.rerun()


def results_panel(S):
    if not (S.retrains or S.events):
        return
    tabs = st.tabs(["Updated models: before vs after", "Event log"] if S.retrains else ["Event log"])
    if S.retrains:
        with tabs[0]:
            t = S.retrain_summary()
            t[["Error before", "Error after"]] = t[["Error before", "Error after"]].round(4)
            t["Change (%)"] = t["Change (%)"].round(1)
            wide(st.dataframe, t, hide_index=True)
            st.caption("Negative change = fewer errors after the update. Early numbers cover few batches.")
    with tabs[-1]:
        wide(st.dataframe, pd.DataFrame(S.events), hide_index=True, height=180)


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if not core.RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install river` and restart.")
        st.stop()
    S = st.session_state.get("S")
    h, p = st.columns([6, 1.3])
    h.markdown("### 📈 Model Drift Dashboard")
    with p:
        how_it_works(S)

    settings_panel(S)
    S = st.session_state.get("S")
    if S is None:
        st.info("Pick the target column above and click **Start monitoring**.")
        st.stop()

    controls(S)
    status_line(S)
    if st.session_state.get("flash"):
        lvl, txt = st.session_state.pop("flash")
        getattr(st, lvl)(txt)

    if S.ready:                                   # chart left, decision right: fits one screen
        with st.spinner("Comparing models on the newest data..."):
            S.ensure_decision()
        left, right = st.columns([1.25, 1])
        with left:
            drift_chart(S, 380)
        with right:
            decision_panel(S)
    else:
        if S.k:
            drift_chart(S, 340)
        if S.alert and S.alert.status == "waiting":
            st.progress(min(S.new_rows / S.cfg.min_rows, 1.0),
                        text=f"Possible drift since batch {S.alert.start + 1}. Collecting new labelled data: "
                             f"{S.new_rows:,} / {S.cfg.min_rows:,} rows")
    results_panel(S)


if __name__ == "__main__":
    main()
