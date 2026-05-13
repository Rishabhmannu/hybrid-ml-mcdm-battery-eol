"""Section 1 — Cell trajectory visualisation (Plotly).

Three stacked panels for the selected cell:
  1. SoH (%) vs cycle  — aging trajectory
  2. Voltage / current / temperature ranges vs cycle  — operating envelope
  3. dQ/dV peak position vs cycle  — chemistry-fingerprint feature

Every panel draws a vertical reference line at the current cycle so the user
sees where the prediction below was taken from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_AXIS_LABEL = dict(showgrid=True, gridcolor="#EEE", zeroline=False)
_FIG_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=30, b=30),
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
    showlegend=False,
)


def _vline(fig: go.Figure, cycle_n: int, n_rows: int = 1) -> None:
    """Draw a vertical dashed line at `cycle_n` across all rows of a subplot fig."""
    for row in range(1, n_rows + 1):
        fig.add_vline(
            x=cycle_n,
            line=dict(color="#1A1A1A", dash="dot", width=1),
            row=row, col=1,
        )


def _soh_plot(cell_df: pd.DataFrame, cycle_n: int) -> go.Figure:
    soh_pct = cell_df["soh"].clip(lower=0, upper=1.5) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cell_df["cycle"], y=soh_pct,
        mode="lines",
        line=dict(color="#1A1A1A", width=1.5),
        name="SoH",
        hovertemplate="cycle %{x}<br>SoH %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="EoL (SoH=80%)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#888"))
    fig.update_layout(
        title="State of Health vs cycle",
        xaxis=dict(title="cycle", **_AXIS_LABEL),
        yaxis=dict(title="SoH (%)", **_AXIS_LABEL),
        height=280,
        **_FIG_LAYOUT,
    )
    fig.add_vline(x=cycle_n, line=dict(color="#1A1A1A", dash="dot", width=1))
    return fig


def _has_signal(series: pd.Series) -> bool:
    """A column is informative if it has at least 1% relative variation across cycles."""
    s = series.dropna()
    if s.empty:
        return False
    rng = float(s.max() - s.min())
    base = max(abs(float(s.mean())), 1e-6)
    return rng / base > 0.01


def _has_temp_signal(cell_df: pd.DataFrame) -> bool:
    """Temperature is real if t_mean has non-NaN values that aren't all zeros
    (the unify pipeline median-imputes missing temps to 0 for some sources)."""
    t = cell_df["t_mean"].dropna()
    return (not t.empty) and (t.abs().max() > 0.5)


def _vit_plot(cell_df: pd.DataFrame, cycle_n: int) -> go.Figure | None:
    """Build a multi-panel V/I/T figure, skipping panels with no informative signal."""
    v_mean_signal = _has_signal(cell_df["v_mean"])
    i_mean_signal = _has_signal(cell_df["i_mean"])
    t_signal = _has_temp_signal(cell_df)

    panels = []  # list of (title, build_fn) where build_fn(fig, row) adds traces
    if v_mean_signal:
        panels.append(("Voltage (V)", "voltage"))
    if i_mean_signal:
        panels.append(("Current (A)", "current"))
    if t_signal:
        panels.append(("Temperature (°C)", "temperature"))

    if not panels:
        return None

    titles = [t for t, _ in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=titles)

    for row, (_, kind) in enumerate(panels, start=1):
        if kind == "voltage":
            # Use min/max range band only when min/max actually move across cycles —
            # otherwise they're protocol cutoffs and the wide constant band misleads.
            spread = _has_signal(cell_df["v_max"]) or _has_signal(cell_df["v_min"])
            if spread:
                fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["v_max"],
                                         mode="lines", line=dict(color="#444", width=1),
                                         name="v_max", showlegend=False), row=row, col=1)
                fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["v_min"],
                                         mode="lines", line=dict(color="#999", width=1),
                                         fill="tonexty",
                                         fillcolor="rgba(150,150,150,0.2)",
                                         name="v_min", showlegend=False), row=row, col=1)
            fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["v_mean"],
                                     mode="lines", line=dict(color="#1A1A1A", width=1.5),
                                     name="v_mean", showlegend=False), row=row, col=1)
        elif kind == "current":
            spread = _has_signal(cell_df["i_max"]) or _has_signal(cell_df["i_min"])
            if spread:
                fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["i_max"],
                                         mode="lines", line=dict(color="#444", width=1),
                                         name="i_max", showlegend=False), row=row, col=1)
                fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["i_min"],
                                         mode="lines", line=dict(color="#999", width=1),
                                         fill="tonexty",
                                         fillcolor="rgba(150,150,150,0.2)",
                                         name="i_min", showlegend=False), row=row, col=1)
            fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["i_mean"],
                                     mode="lines", line=dict(color="#1A1A1A", width=1.5),
                                     name="i_mean", showlegend=False), row=row, col=1)
        elif kind == "temperature":
            fig.add_trace(go.Scatter(x=cell_df["cycle"], y=cell_df["t_mean"],
                                     mode="lines", line=dict(color="#1A1A1A", width=1.5),
                                     name="t_mean", showlegend=False), row=row, col=1)

    fig.update_xaxes(title_text="cycle", row=len(panels), col=1, **_AXIS_LABEL)
    for r in range(1, len(panels) + 1):
        fig.update_yaxes(**_AXIS_LABEL, row=r, col=1)
    fig.update_layout(height=110 * len(panels) + 80, **_FIG_LAYOUT)
    _vline(fig, cycle_n, n_rows=len(panels))
    return fig


def _dqdv_plot(cell_df: pd.DataFrame, cycle_n: int) -> go.Figure | None:
    """Plot dQ/dV peak position vs cycle, charge + discharge. None if all NaN."""
    has_charge = cell_df["v_peak_dqdv_charge"].notna().any()
    has_disch = cell_df["v_peak_dqdv_discharge"].notna().any()
    if not (has_charge or has_disch):
        return None

    fig = go.Figure()
    if has_charge:
        fig.add_trace(go.Scatter(
            x=cell_df["cycle"], y=cell_df["v_peak_dqdv_charge"],
            mode="lines", name="charge peak",
            line=dict(color="#1A1A1A", width=1.5),
            hovertemplate="cycle %{x}<br>V_peak (charge) %{y:.3f} V<extra></extra>",
        ))
    if has_disch:
        fig.add_trace(go.Scatter(
            x=cell_df["cycle"], y=cell_df["v_peak_dqdv_discharge"],
            mode="lines", name="discharge peak",
            line=dict(color="#888", width=1.5, dash="dash"),
            hovertemplate="cycle %{x}<br>V_peak (discharge) %{y:.3f} V<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="dQ/dV peak voltage vs cycle  (chemistry-fingerprint feature)",
                   y=0.97, yanchor="top"),
        xaxis=dict(title="cycle", **_AXIS_LABEL),
        yaxis=dict(title="V_peak (V)", **_AXIS_LABEL),
        height=280,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.02,
                    yanchor="bottom", font=dict(size=10)),
        margin=dict(l=40, r=20, t=70, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="sans-serif", size=12, color="#1A1A1A"),
    )
    fig.add_vline(x=cycle_n, line=dict(color="#1A1A1A", dash="dot", width=1))
    return fig


def render(cell_df: pd.DataFrame, cycle_n: int) -> None:
    with st.container(border=True):
        st.subheader("1 · Cell trajectory")
        st.caption(
            "How this cell has aged across the cycles you can see in the dataset. "
            "The dotted vertical line marks the current cycle the prediction below was taken at."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_Trajectory plots are the raw evidence — every ML model below takes "
            "these per-cycle patterns as input. **State of Health (SoH)** = current "
            "capacity ÷ original capacity. End-of-life (EoL) is conventionally "
            "defined as SoH = 80 %._"
            "</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_soh_plot(cell_df, cycle_n), width="stretch")
        vit = _vit_plot(cell_df, cycle_n)
        if vit is not None:
            st.plotly_chart(vit, width="stretch")
        else:
            st.caption(
                "_Voltage / current / temperature per-cycle aggregates show no "
                "informative variation for this cell — its source dataset only "
                "logged protocol-fixed values per cycle._"
            )
        dqdv = _dqdv_plot(cell_df, cycle_n)
        if dqdv is not None:
            st.plotly_chart(dqdv, width="stretch")
        else:
            st.caption(
                "_dQ/dV features unavailable for this cell — the source dataset "
                "did not export per-cycle voltage curves._"
            )
