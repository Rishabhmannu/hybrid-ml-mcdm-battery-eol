"""Section 4 — Remaining useful life.

Side-by-side predictions from two RUL models trained on the uncensored
audited corpus (the only configuration that passed the 2% test-RMSE-of-range
gate, per Iter-3 §3.12):

  · XGBoost RUL (audited + uncensored)   ←  production, 1.92% RMSE-of-range
  · TCN RUL      (audited + uncensored)   ←  DL secondary, 2.23%

Then a trajectory chart shows the cell's observed SoH up to the current
cycle, with each model's projected end-of-life line (SoH = 80 %) drawn out
into the future for direct visual comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import xgboost as xgb

from frontend.lib.featurize_single_cell import (
    featurize_single_cell,
    prepare_tcn_sequence,
)
from frontend.lib.hf_resolve import hf_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Metrics file stays in the GitHub repo; weights and scalers live on HF.
TCN_METRICS = PROJECT_ROOT / "results" / "tables" / "tcn_rul" / "metrics.json"

_AXIS = dict(showgrid=True, gridcolor="#EEE", zeroline=False)
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=40, r=20, t=30, b=30),
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
)


@st.cache_resource
def _load_xgb_rul():
    model = xgb.XGBRegressor()
    model.load_model(hf_path("xgboost_rul/xgboost_rul_audited_uncensored.json"))
    scaler = joblib.load(hf_path("xgboost_rul/feature_scaler_audited_uncensored.pkl"))
    return model, scaler


@st.cache_resource
def _load_tcn():
    from src.models.tcn_rul import BatteryTCN
    from src.utils.config import TCN_CONFIG
    metrics = json.loads(TCN_METRICS.read_text())
    seq_len = metrics.get("seq_len", 60)
    model = BatteryTCN(
        input_size=metrics["n_features"],
        num_channels=TCN_CONFIG["num_channels"],
        kernel_size=TCN_CONFIG["kernel_size"],
        dropout=TCN_CONFIG["dropout"],
    )
    ckpt = torch.load(hf_path("tcn_rul/best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = joblib.load(hf_path("tcn_rul/feature_scaler.pkl"))
    return model, scaler, seq_len, metrics


def _xgb_rul_predict(model, scaler, cell_df: pd.DataFrame, cycle_n: int) -> float:
    x = featurize_single_cell(cell_df, cycle_n, scaler, mode="audited")
    return max(0.0, float(model.predict(x)[0]))


def _tcn_rul_predict(model, scaler, seq_len: int,
                     cell_df: pd.DataFrame, cycle_n: int) -> float:
    seq = prepare_tcn_sequence(cell_df, cycle_n, scaler, seq_len=seq_len)
    with torch.no_grad():
        pred = float(model(torch.from_numpy(seq).float()).item())
    return max(0.0, pred)


def _trajectory_with_projection(
    cell_df: pd.DataFrame,
    cycle_n: int,
    current_soh_pct: float | None,
    rul_xgb: float,
    rul_tcn: float,
) -> go.Figure:
    """Plot observed SoH(%) curve up to cycle_n with dashed projections to EoL=80%."""
    soh_pct = cell_df["soh"].clip(lower=0, upper=1.5) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cell_df["cycle"], y=soh_pct,
        mode="lines", name="observed SoH",
        line=dict(color="#1A1A1A", width=1.5),
        hovertemplate="cycle %{x}<br>SoH %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="EoL (SoH=80%)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#888"))
    fig.add_vline(x=cycle_n, line=dict(color="#1A1A1A", dash="dot", width=1))

    if current_soh_pct is not None and not np.isnan(current_soh_pct):
        for rul, label, color in [
            (rul_xgb, "XGBoost projection", "#3366CC"),
            (rul_tcn, "TCN projection",     "#CC6633"),
        ]:
            if rul <= 0:
                continue
            eol_cycle = cycle_n + rul
            fig.add_trace(go.Scatter(
                x=[cycle_n, eol_cycle],
                y=[current_soh_pct, 80.0],
                mode="lines+markers", name=label,
                line=dict(color=color, dash="dash", width=1.5),
                marker=dict(size=[0, 8], color=color),
                hovertemplate=f"{label}<br>cycle %{{x:.0f}}<br>SoH %{{y:.2f}}%<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="SoH trajectory + RUL projection to EoL",
                   y=0.97, yanchor="top"),
        xaxis=dict(title="cycle", **_AXIS),
        yaxis=dict(title="SoH (%)", **_AXIS),
        height=340, showlegend=True,
        legend=dict(orientation="h", x=0, y=1.04,
                    yanchor="bottom", font=dict(size=10)),
        margin=dict(l=40, r=20, t=70, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="sans-serif", size=12, color="#1A1A1A"),
    )
    return fig


def render(cell_df: pd.DataFrame, cycle_n: int) -> dict:
    with st.container(border=True):
        st.subheader("4 · Remaining useful life")
        st.caption(
            "How many more cycles until this cell hits End-of-Life (SoH = 80 %). "
            "Two models predict in parallel — a classical machine-learning model "
            "and a deep-learning model — so you can see where they agree and "
            "where they disagree."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_**Remaining Useful Life (RUL)** is the number of cycles a cell can "
            "still deliver before its capacity drops below the End-of-Life "
            "threshold. **XGBoost** is a tree-based machine-learning method; "
            "**TCN** (Temporal Convolutional Network) is a deep-learning model "
            "that reads the cell's recent cycle history._"
            "</span>",
            unsafe_allow_html=True,
        )

        xgb_model, xgb_scaler = _load_xgb_rul()
        tcn_model, tcn_scaler, tcn_seq_len, _ = _load_tcn()

        rul_xgb = _xgb_rul_predict(xgb_model, xgb_scaler, cell_df, cycle_n)
        rul_tcn = _tcn_rul_predict(tcn_model, tcn_scaler, tcn_seq_len, cell_df, cycle_n)

        row = cell_df.iloc[(cell_df["cycle"] - cycle_n).abs().idxmin()]
        current_soh_pct = (float(row["soh"]) * 100.0) if pd.notna(row["soh"]) else None

        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost RUL", f"{rul_xgb:,.0f} cycles",
                  help=("Machine-learning model (gradient-boosted trees). "
                       "Looks at a single cycle's features and predicts how "
                       "many more cycles remain. Typically accurate to within "
                       "~2 % of the prediction range on held-out test cells."))
        c2.metric("TCN RUL", f"{rul_tcn:,.0f} cycles",
                  help=("Deep-learning model (Temporal Convolutional Network). "
                       "Reads the cell's most recent 60-cycle history rather "
                       "than a single snapshot. Slightly less accurate than "
                       "the ML model but a useful second opinion."))
        delta = rul_xgb - rul_tcn
        agree_word = ("good" if abs(delta) < 50
                      else "modest" if abs(delta) < 200
                      else "poor")
        c3.metric("Model agreement", agree_word,
                  delta=f"Δ {delta:+,.0f} cycles", delta_color="off",
                  help="Distance between predictions: <50 cycles = good, 50–200 = modest, >200 = poor")

        st.plotly_chart(
            _trajectory_with_projection(cell_df, cycle_n, current_soh_pct, rul_xgb, rul_tcn),
            width="stretch",
        )

        if current_soh_pct is not None and current_soh_pct < 80:
            st.markdown(
                f"This cell is **already past EoL** at cycle {cycle_n} "
                f"(observed SoH {current_soh_pct:.2f} %). Both models should "
                "converge near 0 here — any large positive RUL is the model "
                "extrapolating outside its training distribution."
            )
        else:
            st.markdown(
                f"XGBoost predicts **~{rul_xgb:,.0f} cycles** until EoL; "
                f"TCN predicts **~{rul_tcn:,.0f}**. "
                f"The two models {'agree closely' if abs(delta) < 50 else 'disagree by ' + f'{abs(delta):,.0f} cycles'}."
            )

    return {"rul_xgb": rul_xgb, "rul_tcn": rul_tcn,
            "current_soh_pct": current_soh_pct}
