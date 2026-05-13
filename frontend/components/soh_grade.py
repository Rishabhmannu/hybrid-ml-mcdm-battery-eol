"""Section 3 — Health assessment.

The richest panel of the demo. For the selected cell + cycle:

  · Global XGBoost SoH (audited) vs Chemistry-Router specialist prediction
  · Observed SoH side-by-side
  · Chemistry tag + specialist provenance + grade-acc
  · A horizontal grade ladder (A > 80 % > B > 60 % > C > 40 % > D) with a
    marker at the specialist's predicted SoH
  · SHAP top-5 contributions on the global model
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from frontend.lib.featurize_single_cell import featurize_single_cell
from frontend.lib.hf_resolve import hf_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_AXIS = dict(showgrid=False, zeroline=False)
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=20, r=20, t=30, b=20),
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
)

GRADE_BANDS = [  # (upper SoH %, label, colour)
    (100, "A — second-life ready",       "#3CB371"),
    (80,  "B — refurbishable",           "#F0AD4E"),
    (60,  "C — limited reuse / repair",  "#E07A5F"),
    (40,  "D — recycling-only",          "#B33B3B"),
]


@st.cache_resource
def _load_global_soh():
    model = xgb.XGBRegressor()
    model.load_model(hf_path("xgboost_soh/xgboost_soh_audited.json"))
    scaler = joblib.load(hf_path("xgboost_soh/feature_scaler_audited.pkl"))
    return model, scaler


@st.cache_resource
def _load_router():
    """Build a ChemistryRouter from HF-hosted artefacts.

    ChemistryRouter.load() expects a local manifest pointing at project-relative
    file paths. We instead construct the dataclass directly so each submodel
    resolves through the HF cache.
    """
    from src.models.chemistry_router import ChemistryRouter

    manifest = json.loads(Path(hf_path("per_chemistry/router_manifest.json")).read_text())
    models: dict = {}
    for entry in manifest["chemistries"]:
        # entry["model_path"] is "models/per_chemistry/<chem>/xgboost_soh_audited.json"
        hf_filename = entry["model_path"].split("models/", 1)[-1]
        m = xgb.XGBRegressor()
        m.load_model(hf_path(hf_filename))
        models[entry["chemistry"]] = m

    fallback = None
    fb = manifest.get("fallback_global_model")
    if fb:
        # Resolve the audited global SoH from HF as the fallback
        fb_filename = fb.split("models/", 1)[-1]
        try:
            fallback = xgb.XGBRegressor()
            fallback.load_model(hf_path(fb_filename))
        except Exception:
            fallback = None

    feature_names = json.loads(Path(hf_path("per_chemistry/feature_names.json")).read_text())
    router = ChemistryRouter(
        models=models,
        fallback_model=fallback,
        feature_names=feature_names,
        manifest=manifest,
    )
    scaler = joblib.load(hf_path("per_chemistry/feature_scaler.pkl"))
    chem_stats = {e["chemistry"]: e for e in manifest["chemistries"]}
    return router, scaler, chem_stats


@st.cache_resource
def _load_shap_explainer():
    import shap
    model, _ = _load_global_soh()
    return shap.TreeExplainer(model)


@st.cache_data
def _load_feature_names() -> list[str]:
    return json.loads(Path(hf_path("per_chemistry/feature_names.json")).read_text())


def _grade_from_soh(soh_pct: float) -> str:
    if soh_pct > 80: return "A"
    if soh_pct > 60: return "B"
    if soh_pct > 40: return "C"
    return "D"


def _bar_comparison(observed: float | None, global_pred: float, specialist_pred: float) -> go.Figure:
    labels, vals, colors = [], [], []
    if observed is not None and not np.isnan(observed):
        labels.append("Observed");        vals.append(observed);        colors.append("#999")
    labels.append("Global model");        vals.append(global_pred);     colors.append("#444")
    labels.append("Specialist");          vals.append(specialist_pred); colors.append("#1A1A1A")

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.add_vline(x=80, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="EoL", annotation_position="top",
                  annotation_font=dict(size=10, color="#888"))
    fig.update_layout(
        title="SoH at this cycle",
        xaxis=dict(title="SoH (%)", range=[0, max(105, max(vals) + 8)], **_AXIS),
        yaxis=_AXIS, height=200, showlegend=False, **_LAYOUT,
    )
    return fig


def _grade_ladder(predicted_soh: float, predicted_grade: str) -> go.Figure:
    fig = go.Figure()
    for upper, label, colour in GRADE_BANDS:
        lower = upper - 20 if upper != 40 else 0
        fig.add_shape(
            type="rect", x0=lower, x1=upper, y0=0, y1=1,
            line=dict(width=0), fillcolor=colour, opacity=0.35,
        )
        fig.add_annotation(
            x=(lower + upper) / 2, y=0.5, text=label,
            showarrow=False, font=dict(size=11, color="#1A1A1A"),
        )
    fig.add_vline(x=predicted_soh, line=dict(color="#1A1A1A", width=2),
                  annotation_text=f"this cell: grade {predicted_grade}  ({predicted_soh:.2f}%)",
                  annotation_position="top",
                  annotation_font=dict(size=12, color="#1A1A1A"))
    fig.update_layout(
        title="Grade ladder",
        xaxis=dict(range=[0, 100], title="SoH (%)", **_AXIS),
        yaxis=dict(range=[0, 1], showticklabels=False, **_AXIS),
        height=140, showlegend=False, **_LAYOUT,
    )
    return fig


def _shap_bar(shap_vals: np.ndarray, feature_names: list[str], top_n: int = 5) -> go.Figure:
    order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = shap_vals[order][::-1]
    colors = ["#3CB371" if v > 0 else "#B33B3B" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line=dict(color="#888", width=1))
    fig.update_layout(
        title=f"Top {top_n} SHAP contributions (this cell, global model)",
        xaxis=dict(title="SHAP value (pp shifted from baseline)", **_AXIS),
        yaxis=_AXIS, height=240, showlegend=False, **_LAYOUT,
    )
    return fig


def render(cell_df: pd.DataFrame, cycle_n: int) -> dict:
    with st.container(border=True):
        st.subheader("3 · Health assessment")
        st.caption(
            "Two parallel predictions of this cell's State of Health — one from "
            "a general-purpose model trained on every chemistry, one from a "
            "specialist trained only on cells of this cell's chemistry. The "
            "specialist tends to be more accurate, especially for less-common "
            "chemistries. Bottom panel: which features drove the prediction."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_**SoH** (State of Health) is the headline number. The **grade "
            "ladder** converts SoH into a routing category A/B/C/D. **SHAP** "
            "shows how each feature pushed this cell's prediction up (green) or "
            "down (red) vs the model's baseline expectation._"
            "</span>",
            unsafe_allow_html=True,
        )

        global_model, global_scaler = _load_global_soh()
        router, router_scaler, chem_stats = _load_router()
        feature_names = _load_feature_names()

        x_global = featurize_single_cell(cell_df, cycle_n, global_scaler, mode="audited")
        x_router = featurize_single_cell(cell_df, cycle_n, router_scaler, mode="audited")

        global_pred = float(global_model.predict(x_global)[0])
        chemistry = str(cell_df["chemistry"].iloc[0])
        specialist_pred = float(router.predict_soh(x_router, [chemistry])[0])

        row = cell_df.iloc[(cell_df["cycle"] - cycle_n).abs().idxmin()]
        observed = (float(row["soh"]) * 100.0) if pd.notna(row["soh"]) else None

        predicted_grade = _grade_from_soh(specialist_pred)

        st.plotly_chart(_bar_comparison(observed, global_pred, specialist_pred),
                        width="stretch")

        stats = chem_stats.get(chemistry)
        if stats:
            n_tr = stats["n_train"]
            ga = stats["test_grade_acc"]
            st.markdown(
                f"Cell tagged as **{chemistry}** — routed to the {chemistry} specialist "
                f"(trained on {n_tr:,} cycles · test grade-acc {ga:.2%}). "
                f"Specialist says SoH = **{specialist_pred:.2f}%** → grade **{predicted_grade}**."
            )
        else:
            st.warning(
                f"No specialist for chemistry **{chemistry}** — using global "
                f"model fallback. Specialist coverage: {sorted(chem_stats)}."
            )

        st.plotly_chart(_grade_ladder(specialist_pred, predicted_grade), width="stretch")

        shap_vals = None
        try:
            explainer = _load_shap_explainer()
            shap_vals = explainer.shap_values(x_global)[0]
            st.plotly_chart(_shap_bar(shap_vals, feature_names), width="stretch")
        except Exception as exc:
            st.info(f"SHAP explanation unavailable: {exc}", icon=None)

    return {
        "global_pred": global_pred,
        "specialist_pred": specialist_pred,
        "grade": predicted_grade,
        "chemistry": chemistry,
        "observed": observed,
        "shap_values": shap_vals,
        "feature_names": feature_names,
    }
