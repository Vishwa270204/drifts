import pandas as pd
import streamlit as st

from drift import Session

from .helpers import wide


def retrain_results(S: Session):
    st.subheader("Updated models: before vs after")
    t = S.retrain_summary()
    for c in ("Error before", "Error after"):
        t[c] = t[c].round(4)
    t["Change (%)"] = t["Change (%)"].round(1)
    wide(st.dataframe, t, hide_index=True)
    st.caption("Negative change = fewer errors after the update. Early numbers cover few batches.")


def event_log(S: Session):
    if S.events:
        with st.expander("Event log"):
            wide(st.dataframe, pd.DataFrame(S.events), hide_index=True)
