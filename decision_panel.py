import numpy as np
import streamlit as st

from drift import KEEP, UPDATE, Session, fmt_score

from .helpers import flash


def decision_panel(S: Session):
    d = S.cand
    with st.container(border=True):
        st.subheader("⚠️ Drift detected")
        if d.error:
            st.error(d.error)
            if st.button("Keep current model", key="skip_err"):
                flash("info", S.keep("not enough data to compare"))
                st.rerun()
            return

        st.write(f"Error started rising at **batch {S.alert.start + 1}**. "
                 f"Both models were tested on the **{d.hold}** most recent labelled rows "
                 f"(lower {d.metric.lower()} is better).")
        can_update = UPDATE in d.scores and np.isfinite(d.scores[UPDATE])
        a, b = st.columns(2)
        a.metric("Current model", fmt_score(d.metric, d.scores[KEEP]))
        if can_update:
            ch = d.change_pct(UPDATE)
            b.metric("Updated model", fmt_score(d.metric, d.scores[UPDATE]),
                     delta=None if np.isnan(ch) else f"{ch:+.1f}%", delta_color="inverse")
        else:
            b.metric("Updated model", "n/a")

        if d.rec == UPDATE:
            st.success("**Recommended: update the model.** It does better on recent data.")
        else:
            st.info("**Recommended: keep the current model.** The update is not better on recent data.")
        if d.notes:
            st.caption("⚠️ Take with care: " + "; ".join(d.notes) + ".")

        rec_update = d.rec == UPDATE
        c1, c2, _ = st.columns([1.4, 1.4, 2.2])
        if c1.button("Update model", type="primary" if rec_update else "secondary", disabled=not can_update,
                     key=f"apply_{S.k}"):
            ok, msg = S.apply(UPDATE)
            flash("success" if ok else "error", msg)
            st.rerun()
        if c2.button("Keep current model", type="secondary" if rec_update else "primary", key=f"keep_{S.k}"):
            flash("info", S.keep())
            st.rerun()
