"""Section 5 — MCDM routing decision.

Given the predicted grade from Section 3 and the regulatory regime chosen in
the sidebar, this panel renders the canonical 6-criterion Fuzzy-BWM + TOPSIS
ranking of the 4 end-of-life routes:

    Grid-scale ESS · Home/Distributed ESS · Component Reuse · Direct Recycling

The RQ2 sub-panel at the bottom shows the top recommendation under all five
regulatory regimes side-by-side — the headline result of the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.mcdm.sensitivity import _resolve_scenarios
from src.mcdm.topsis import (
    CANONICAL_ALTERNATIVES,
    CANONICAL_CRITERIA,
    CANONICAL_TYPES,
    build_canonical_decision_matrix,
    topsis_rank,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AXIS = dict(showgrid=False, zeroline=False)
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=40, r=20, t=30, b=20),
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
)


@st.cache_resource
def _load_scenarios() -> dict[str, np.ndarray]:
    return _resolve_scenarios()


def _decision_matrix_table(grade: str) -> pd.DataFrame:
    dm = build_canonical_decision_matrix(grade)
    df = pd.DataFrame(dm, index=CANONICAL_ALTERNATIVES, columns=CANONICAL_CRITERIA)
    df.index.name = "Route"
    return df


def _styled_dm(df: pd.DataFrame):
    """Cell shading: benefit columns higher-is-greener, cost columns higher-is-redder."""
    styler = df.style.format("{:.1f}")
    for col, ctype in zip(CANONICAL_CRITERIA, CANONICAL_TYPES):
        cmap = "Greens" if ctype == "benefit" else "Reds"
        styler = styler.background_gradient(subset=[col], cmap=cmap,
                                            vmin=1.0, vmax=10.0)
    return styler


def _weights_bar(weights: np.ndarray, regime: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=weights, y=CANONICAL_CRITERIA, orientation="h",
        marker=dict(color="#1A1A1A"),
        text=[f"{w:.3f}" for w in weights], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=f"Criterion weights · {regime}",
        xaxis=dict(title="weight", range=[0, max(weights) * 1.25], **_AXIS),
        yaxis=_AXIS, height=240, showlegend=False, **_LAYOUT,
    )
    return fig


def _ranking_bar(ranked: pd.DataFrame) -> go.Figure:
    ordered = ranked.sort_values("closeness", ascending=True)
    colors = ["#3CB371" if r == 1 else "#888" for r in ordered["rank"]]
    fig = go.Figure(go.Bar(
        x=ordered["closeness"], y=ordered["alternative"], orientation="h",
        marker=dict(color=colors),
        text=[f"rank {r}  ·  {c:.4f}" for r, c in zip(ordered["rank"], ordered["closeness"])],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title="TOPSIS closeness coefficient (higher = better route)",
        xaxis=dict(title="closeness", range=[0, ordered["closeness"].max() * 1.20], **_AXIS),
        yaxis=_AXIS, height=240, showlegend=False, **_LAYOUT,
    )
    return fig


def _run_topsis(grade: str, weights: np.ndarray) -> pd.DataFrame:
    dm = build_canonical_decision_matrix(grade)
    res = topsis_rank(dm, weights, CANONICAL_TYPES)
    return pd.DataFrame({
        "alternative": CANONICAL_ALTERNATIVES,
        "closeness": res["closeness"],
        "rank": res["ranking"],
        "d_positive": res["d_positive"],
        "d_negative": res["d_negative"],
    })


def render(grade: str, regime: str) -> dict:
    with st.container(border=True):
        st.subheader("5 · MCDM routing decision")
        st.caption(
            "Canonical Fuzzy-BWM + TOPSIS over six criteria (SoH, residual Value, "
            "Carbon footprint, Compliance fit, Safety risk, EPR Return). "
            "Carbon and Safety are cost-type and inverted at scoring time. "
            "Switch the regulatory regime in the sidebar to see how the "
            "recommendation flips — that flip *is* RQ2 of the paper."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_**MCDM** (Multi-Criteria Decision-Making) ranks the 4 end-of-life "
            "routes — Grid-scale ESS, Home/Distributed ESS, Component Reuse, "
            "Direct Recycling — using **TOPSIS** (closeness to the ideal point) "
            "and **Fuzzy-BWM** (Best-Worst Method) for the criterion weights. "
            "The closer to 1.0, the better the route._"
            "</span>",
            unsafe_allow_html=True,
        )

        scenarios = _load_scenarios()
        if regime not in scenarios:
            st.warning(f"Unknown regime: {regime!r}. Falling back to Literature Mean.")
            regime = "Literature Mean"
        weights = scenarios[regime]

        dm_df = _decision_matrix_table(grade)
        st.markdown(
            f"**Decision matrix for grade {grade}** "
            f"_(rows = 4 routes, columns = 6 criteria; "
            "benefit columns shaded green, cost columns red)_"
        )
        st.dataframe(_styled_dm(dm_df), width="stretch")

        col_w, col_r = st.columns(2)
        col_w.plotly_chart(_weights_bar(weights, regime), width="stretch")

        ranked = _run_topsis(grade, weights)
        col_r.plotly_chart(_ranking_bar(ranked), width="stretch")

        top = ranked.loc[ranked["rank"] == 1].iloc[0]
        st.markdown(
            f"Under **{regime}** weights for a grade-**{grade}** cell, "
            f"TOPSIS recommends **{top['alternative']}** "
            f"(closeness coefficient {top['closeness']:.4f})."
        )

        st.markdown("---")
        st.markdown("**RQ2 — recommendation under each regulatory regime**")
        rq2_rows = []
        for r_name, r_weights in scenarios.items():
            r_df = _run_topsis(grade, r_weights)
            r_top = r_df.loc[r_df["rank"] == 1].iloc[0]
            rq2_rows.append({
                "Regime": r_name,
                "Top recommendation": r_top["alternative"],
                "Closeness": round(r_top["closeness"], 4),
                "Second choice": r_df.loc[r_df["rank"] == 2].iloc[0]["alternative"],
            })
        rq2_df = pd.DataFrame(rq2_rows)
        st.dataframe(rq2_df, width="stretch", hide_index=True)

        unique_tops = rq2_df["Top recommendation"].nunique()
        if unique_tops == 1:
            st.caption(
                f"All 5 regimes agree on **{rq2_df.iloc[0]['Top recommendation']}** — "
                f"the recommendation is robust for grade {grade}."
            )
        else:
            st.caption(
                f"Regimes split across **{unique_tops} different top routes** — "
                "this is the regulatory-regime sensitivity RQ2 quantifies."
            )

    return {
        "regime": regime,
        "grade": grade,
        "top_alternative": top["alternative"],
        "closeness": float(top["closeness"]),
        "ranked": ranked.to_dict("records"),
        "weights": {c: float(w) for c, w in zip(CANONICAL_CRITERIA, weights)},
    }
