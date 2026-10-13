from pathlib import Path

import streamlit as st

import drift
from drift import SENS, Config, Session

from .helpers import confirm_box

CSV_PATH = Path(__file__).resolve().parent.parent / "concept_drift_stream.csv"   # keep it next to app.py


@st.cache_data(show_spinner="Reading CSV...")
def _read_bytes(data: bytes):
    return drift.read_csv_bytes(data)


def _load():
    """Default CSV if present, otherwise an uploader. Returns (df, name) or (None, None)."""
    if CSV_PATH.exists():
        return _read_bytes(CSV_PATH.read_bytes()), CSV_PATH.name
    up = st.file_uploader("Upload a CSV (rows in time order)", type="csv")
    return (_read_bytes(up.getvalue()), up.name) if up else (None, None)


def config_panel(S):
    """Three choices only: target, sensitivity, batch size. Everything else is detected or defaulted."""
    with st.expander("Settings", expanded=S is None):
        try:
            df, name = _load()
        except Exception as exc:
            st.error(str(exc))
            return
        if df is None:
            return
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        sens = c2.selectbox("Drift sensitivity", list(SENS), index=1)
        batch = int(c3.number_input("Batch size (rows)", 20, 5000, 100, step=10))
        task, tcol = drift.guess_task(df, target), drift.detect_time_col(df, target)
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
