from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import drift_core as core
from drift_core import KEEP, SENS, UPDATE, Config, Session, fmt_score

CSV_PATH = Path(__file__).resolve().parent / "concept_drift_stream.csv"   # keep it next to app.py
PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]
CHART_H = 270         # full-width error chart
CSS = """<style>
/* page frame: full width, small margins, clear of Streamlit's toolbar */
.block-container{padding:4.5rem 1.5rem 2rem 1.5rem;max-width:100%}
footer,#MainMenu{visibility:hidden}
[data-testid="stVerticalBlock"]{gap:.45rem}

/* headings */
h3{padding:0!important;margin:0!important;font-size:1.2rem!important;line-height:1.5!important}
.app-title{font-size:2.6rem;font-weight:800;line-height:1.25;margin:0 0 .15rem 0;white-space:nowrap}
.app-sub{opacity:.7;font-size:1.05rem;margin:0}
.sec{font-size:.78rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;opacity:.6;margin:.1rem 0 0 0}

/* KPI cards */
[data-testid="stMetric"]{background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.18);
  border-radius:10px;padding:.35rem .8rem}
[data-testid="stMetricLabel"]{opacity:.7}
[data-testid="stMetricValue"]{font-size:1.4rem}

/* misc */
[data-testid="stAlert"]{padding:.45rem .8rem}
[data-testid="stProgress"]{margin-bottom:-.2rem}
[data-testid="stTabs"] [role="tablist"]{margin-bottom:-.2rem}
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
    box = st.expander("Settings", expanded=True) if S is None else st.popover("⚙️ Settings", width="stretch")
    with box:
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


# ------------------------------------------------------------------ run controls, status, KPIs, help
def advance(S, n):
    """Run n batches (None = until a decision). Detector failures are shown instead of crashing the page."""
    try:
        S.run(n)
    except core.DetectorError as exc:
        st.error(f"Drift detector error: {exc}")


def controls(S):
    off = not S.can_advance
    a, b, c, d = st.columns([3, 1.2, 1.4, 1])
    if a.button("▶ Run until a decision is needed", type="primary", disabled=off, key="run_all", width="stretch"):
        advance(S, None)
    if b.button("Next batch", disabled=off, key="run_1", width="stretch"):
        advance(S, 1)
    if c.button("Next 10 batches", disabled=off, key="run_10", width="stretch"):
        advance(S, 10)
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
    st.progress(S.k / S.nb if S.nb else 0.0, text=f"Batch {S.k} of {S.nb}   ·   {S.status}")


def kpi_strip(S):
    errs = np.asarray(S.errs, float)
    pct = S.task == "classification"
    latest = "–" if len(errs) == 0 else (f"{errs[-1]:.1%}" if pct else f"{errs[-1]:.3f}")
    alerts = sum(1 for g in S.signals if g["dir"] == "up")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed model", f"v{S.version}")
    c2.metric("Latest error", latest)
    c3.metric("Drift alerts", alerts)
    c4.metric("Model updates", len(S.retrains))


# ------------------------------------------------------------------ chart
def drift_chart(S, height=CHART_H):
    n = min(S.k, len(S.ver), len(S.errs))
    xs, ver, errs = np.arange(1, n + 1), np.asarray(S.ver)[:n], np.asarray(S.errs, float)[:n]
    pct = S.task == "classification"
    fig = go.Figure()
    for i, v in enumerate(sorted(np.unique(ver))):
        idx = np.flatnonzero(ver == v)
        fig.add_trace(go.Scatter(x=xs[idx], y=errs[idx], mode="lines+markers", name=f"Model v{v}",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=5)))
    lv = sorted(j for j in S.levels if j < n)
    if lv:                                          # ADWIN has no threshold; this is its current "normal" level
        y = np.full(n, np.nan)
        y[lv] = [S.levels[j] for j in lv]
        fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", name="ADWIN normal level", connectgaps=False,
                                 line=dict(dash="dash", color="#c62828", width=1.5, shape="hv")))
    for direction, label, color, symbol in (("up", "ADWIN: error rose", "#c62828", "diamond"),
                                            ("down", "ADWIN: error fell", "#2e7d32", "diamond-open")):
        js = [g["batch"] for g in S.signals if g["dir"] == direction and g["batch"] < n]
        if js:
            fig.add_trace(go.Scatter(x=[j + 1 for j in js], y=errs[js], mode="markers", name=label,
                                     marker=dict(color=color, symbol=symbol, size=13,
                                                 line=dict(width=2, color=color))))
    for r in S.retrains:
        if 1 <= r.applied + 1 <= n:
            fig.add_vline(x=r.applied + 1, line_color="#2e7d32", line_width=1.5,
                          annotation_text=f"v{r.version}", annotation_position="top left")
    if S.alert is not None and 1 <= S.alert.start + 1 <= n:
        fig.add_vrect(x0=S.alert.start + 1, x1=n, fillcolor="#c62828", opacity=0.07, line_width=0,
                      annotation_text="Drift", annotation_position="top left")
    fig.update_layout(height=height, margin=dict(t=28, b=30, l=50, r=10), hovermode="x unified",
                      xaxis_title="Batch", yaxis_title="Error rate" if pct else "MAE",
                      legend=dict(orientation="h", y=1.14, x=0))
    if pct:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    wide(st.plotly_chart, fig)


# ------------------------------------------------------------------ bottom row: decision (left) + history (right)
def decision_panel(S):
    d = S.cand
    with st.container(border=True):
        st.markdown("### ⚠️ Drift detected: your decision")
        if d.error:
            st.error(d.error)
            if st.button("Keep current model", key="skip_err"):
                flash("info", S.keep("not enough data to compare"))
                st.rerun()
            return
        al = S.alert
        st.write(f"ADWIN flagged a rise in error at **batch {al.detected + 1}**"
                 + ("." if al.start == al.detected else f"; the change began around **batch {al.start + 1}**."))
        can_update = UPDATE in d.scores and np.isfinite(d.scores[UPDATE])
        cur = fmt_score(d.metric, d.scores[KEEP])
        new = fmt_score(d.metric, d.scores[UPDATE]) if can_update else "n/a"

        m1, m2 = st.columns(2)
        m1.metric("Current model", cur)
        if can_update:
            ch = d.change_pct(UPDATE)
            m2.metric("Updated model", new, delta=None if np.isnan(ch) else f"{ch:+.1f}%", delta_color="inverse")
        else:
            m2.metric("Updated model", new)
        st.caption(f"{d.metric} on the {d.hold} newest labelled rows (lower is better).")

        rec_update = d.rec == UPDATE
        if rec_update:
            st.success(f"**Recommended: update the model** ({new} vs {cur}).")
        else:
            st.info("**Recommended: keep the current model.** The update is not better on recent data.")
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


def idle_panel(S):
    """Same slot and height as the decision panel, so the layout never jumps."""
    with st.container(border=True):
        st.markdown("### Decision")
        if S.alert:
            st.markdown(f"**Not yet.** Possible drift since batch {S.alert.start + 1}; collecting new labelled data.")
            st.progress(min(S.new_rows / S.cfg.min_rows, 1.0),
                        text=f"{S.new_rows:,} / {S.cfg.min_rows:,} new labelled rows")
        elif S.done:
            st.markdown("**None pending.** All batches have been processed.")
        else:
            st.markdown("**None needed.** Error is within the normal range. You will be asked here if drift is "
                        "confirmed.")


def results_panel(S):
    """Always rendered at the same height as the decision panel."""
    with st.container(border=True):
        st.markdown("### History")
        if not (S.retrains or S.events):
            st.caption("Nothing yet. Drift alerts and model updates will be listed here.")
            return
        tabs = st.tabs(["Model updates: before vs after", "Event log"] if S.retrains else ["Event log"])
        if S.retrains:
            with tabs[0]:
                t = S.retrain_summary()
                t[["Error before", "Error after"]] = t[["Error before", "Error after"]].round(4)
                t["Change (%)"] = t["Change (%)"].round(1)
                wide(st.dataframe, t, hide_index=True, height=min(35 * (len(t) + 1) + 3, 320))
                st.caption("Negative change = fewer errors after the update. Early numbers cover few batches.")
        with tabs[-1]:
            ev = pd.DataFrame(S.events)
            wide(st.dataframe, ev, hide_index=True, height=min(35 * (len(ev) + 1) + 3, 320))


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if not core.RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install river` and restart.")
        st.stop()
    missing = [n for n in ("SCHEMA", "DetectorError") if not hasattr(core, n)]
    if missing:
        st.error("`drift_core.py` is out of date for this `app.py`. Upload the latest `drift_core.py` next to it "
                 "and reboot the app.")
        st.stop()
    S = st.session_state.get("S")
    if S is not None and getattr(S, "SCHEMA", None) != core.SCHEMA:     # session made by an older version of the code
        st.session_state.pop("S", None)
        st.session_state.pop("flash", None)
        S = None
        st.toast("The app was updated, so the previous session was reset. Please start monitoring again.")

    # ---- header
    h, p_set = st.columns([6, 1.4])
    h.markdown('<p class="app-title">📈 Model Drift Dashboard</p>'
               '<p class="app-sub">Watch a deployed model for drift and decide when to update it.</p>',
               unsafe_allow_html=True)
    if S is not None:
        with p_set:
            settings_panel(S)                       # compact popover once monitoring has started
    else:
        settings_panel(None)                        # full panel until the first start
    S = st.session_state.get("S")
    if S is None:
        st.info("Pick the target column above and click **Start monitoring**.")
        st.stop()

    # ---- run controls + progress
    controls(S)
    status_line(S)
    if st.session_state.get("flash"):
        lvl, txt = st.session_state.pop("flash")
        getattr(st, lvl)(txt)

    # ---- KPI strip
    kpi_strip(S)

    # ---- full-width error chart
    st.markdown('<p class="sec">Model error over time</p>', unsafe_allow_html=True)
    if S.k:
        drift_chart(S)
    else:
        with st.container(border=True, height=CHART_H):
            st.caption("The error chart appears here. Press **Run** or **Next batch**.")

    # ---- bottom row: decision | history (equal height, always aligned)
    left, right = st.columns(2)
    with left:
        if S.ready:
            with st.spinner("Comparing models on the newest data..."):
                S.ensure_decision()
            decision_panel(S)
        else:
            idle_panel(S)
    with right:
        results_panel(S)


if __name__ == "__main__":
    main()
