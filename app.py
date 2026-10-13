import pandas as pd
import streamlit as st

import drift
from ui.charts import drift_chart
from ui.config_panel import config_panel
from ui.decision_panel import decision_panel
from ui.helpers import CSS, confirm_box, show_flash
from ui.results import event_log, retrain_results


def controls(S):
    off = not S.can_advance
    a, b, c, d = st.columns([2.6, 1, 1.2, 0.8])
    if a.button("▶ Run until a decision is needed", type="primary", disabled=off, key="run_all"):
        S.run(None)
    if b.button("Next batch", disabled=off, key="run_1"):
        S.run(1)
    if c.button("Next 10 batches", disabled=off, key="run_10"):
        S.run(10)
    if d.button("Reset", key="reset"):
        if S.k > 0:
            st.session_state["confirm_reset"] = True
        else:
            st.session_state.pop("S")
            st.rerun()
    if confirm_box("confirm_reset", "Reset the session? Model versions and history will be discarded.", "Yes, reset"):
        st.session_state.pop("S")
        st.session_state.pop("flash", None)
        st.rerun()


def status_row(S):
    a, b, c = st.columns([3, 1, 1.4])
    a.progress(S.k / S.nb if S.nb else 0.0, text=f"Batch {S.k} of {S.nb}")
    b.metric("Model", f"v{S.version}")
    c.metric("Status", S.status)


def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 Model Drift Dashboard")
    if not drift.RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install river` and restart.")
        st.stop()

    config_panel(st.session_state.get("S"))
    S = st.session_state.get("S")
    if S is None:
        st.info("Pick the target column above and click **Start monitoring**.")
        st.stop()

    controls(S)
    status_row(S)
    show_flash()
    if S.k:
        drift_chart(S)

    if S.alert and S.alert.status == "waiting":
        st.progress(min(S.new_rows / S.cfg.min_rows, 1.0),
                    text=f"Possible drift since batch {S.alert.start + 1}. Collecting new labelled data: "
                         f"{S.new_rows:,} / {S.cfg.min_rows:,} rows")
    if S.ready:
        with st.spinner("Comparing models on the newest data..."):
            S.ensure_decision()
        decision_panel(S)

    if S.retrains:
        retrain_results(S)
    event_log(S)


if __name__ == "__main__":
    main()
