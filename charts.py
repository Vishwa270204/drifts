import numpy as np
import plotly.graph_objects as go
import streamlit as st

from drift import Session

from .helpers import wide

PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]


def drift_chart(S: Session):
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
    fig.update_layout(height=320, margin=dict(t=20, b=40, l=50, r=20), hovermode="x unified",
                      xaxis_title="Batch", yaxis_title="Error rate" if pct else "MAE",
                      legend=dict(orientation="h", y=-0.25))
    if pct:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    wide(st.plotly_chart, fig)
