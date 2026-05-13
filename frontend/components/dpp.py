"""Section 6 — Digital Product Passport output.

Builds a unified-schema-compliant DPP from the upstream prediction outputs,
validates it against `data/processed/dpp/unified_dpp_schema.json`, and renders:

  · Tab 1 — human-readable summary card (key fields formatted per category)
  · Tab 2 — raw JSON with copy + download
  · Per-category coverage bar chart
  · Schema validation status
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dpp.schema_mapper import build_dpp, validate_against_schema

_AXIS = dict(showgrid=False, zeroline=False)
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=40, r=20, t=30, b=20),
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
)

DPP_CATEGORIES = [
    "identity",
    "chemistry_and_composition",
    "performance_and_durability",
    "state_of_health",
    "carbon_footprint",
    "supply_chain_due_diligence",
    "circularity_and_eol",
    "labels_and_compliance",
    "dismantling_and_safety",
]


def _category_coverage(dpp: dict) -> pd.DataFrame:
    def has_value(v):
        if v is None: return False
        if isinstance(v, str): return v not in ("", "placeholder")
        if isinstance(v, list): return len(v) > 0
        if isinstance(v, dict): return any(has_value(x) for x in v.values())
        return True

    rows = []
    for cat in DPP_CATEGORIES:
        block = dpp.get(cat, {})
        if isinstance(block, dict) and block:
            leaves = list(block.values())
            populated = sum(1 for v in leaves if has_value(v))
            rows.append({
                "category": cat.replace("_", " "),
                "populated": populated,
                "total": len(leaves),
                "coverage": populated / max(len(leaves), 1),
            })
        else:
            rows.append({"category": cat.replace("_", " "),
                         "populated": 0, "total": 0, "coverage": 0.0})
    return pd.DataFrame(rows)


def _coverage_bar(cov_df: pd.DataFrame) -> go.Figure:
    colors = ["#3CB371" if c >= 0.7
              else "#F0AD4E" if c >= 0.4
              else "#B33B3B"
              for c in cov_df["coverage"]]
    fig = go.Figure(go.Bar(
        x=cov_df["coverage"] * 100,
        y=cov_df["category"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{p}/{t}  ·  {c:.0%}" for p, t, c
              in zip(cov_df["populated"], cov_df["total"], cov_df["coverage"])],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title="DPP field coverage by category",
        xaxis=dict(title="% populated", range=[0, 115], **_AXIS),
        yaxis=dict(autorange="reversed", **_AXIS),
        height=320, showlegend=False, **_LAYOUT,
    )
    return fig


def _render_summary_card(dpp: dict) -> None:
    iden = dpp["identity"]
    chem = dpp["chemistry_and_composition"]
    perf = dpp["performance_and_durability"]
    soh = dpp["state_of_health"]
    circ = dpp["circularity_and_eol"]
    epr = circ.get("epr_compliance", {})
    prov = dpp["provenance"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Identity & chemistry")
        st.markdown(
            f"- **DPP ID**: `{iden['dpp_id']}`\n"
            f"- **Battery ID**: `{iden['battery_id']}`\n"
            f"- **Product status**: {iden['product_status']}\n"
            f"- **Form factor**: {iden['form_factor'] or 'n/a'}\n"
            f"- **Cathode chemistry**: {chem['cathode_chemistry']}\n"
            f"- **Anode chemistry**: {chem['anode_chemistry'] or 'n/a'}\n"
            f"- **Critical raw materials**: "
            f"{', '.join(chem['critical_raw_materials_present']) or 'none'}"
        )
        st.markdown("##### Performance & durability")
        st.markdown(
            f"- **Rated capacity**: {perf['rated_capacity_Ah']} Ah\n"
            f"- **Voltage window**: {perf['voltage_min_V']} V – {perf['voltage_max_V']} V\n"
            f"- **Expected lifetime**: {perf['expected_lifetime_cycles']:,} cycles\n"
            f"- **EoL threshold**: SoH ≤ {perf['capacity_threshold_for_exhaustion_pct']:.0f} %"
        )
    with c2:
        st.markdown("##### State of health")
        rul = soh["rul_remaining_cycles"]
        st.markdown(
            f"- **SoH**: {soh['soh_percent']:.2f} %\n"
            f"- **Cycles completed**: {soh['cycles_completed']:,}\n"
            f"- **RUL remaining**: {f'{int(rul):,} cycles' if rul is not None else 'n/a'}\n"
            f"- **Estimation method**: {soh['estimation_method'][:120]}…\n"
            f"- **Data source**: {soh['data_source']}\n"
            f"- **Last assessed**: {soh['last_assessed']}"
        )
        st.markdown("##### Circularity & EoL route")
        scores = circ.get("all_route_scores") or []
        route_score_str = (f"{circ['route_score']:.4f}"
                           if circ["route_score"] is not None else "n/a")
        recovery_pct = epr.get("recovery_target_pct", "n/a")
        recovery_fy = epr.get("recovery_target_fy", "")
        take_back = epr.get("take_back_route", "n/a")
        st.markdown(
            f"- **Grade**: {circ['grade']}\n"
            f"- **Recommended route**: **{circ['recommended_route']}**\n"
            f"- **Route score** (closeness): {route_score_str}\n"
            f"- **Ranking method**: {circ['route_ranking_method']}\n"
            f"- **Take-back route**: {take_back}\n"
            f"- **BWMR recovery target**: {recovery_pct} % ({recovery_fy})\n"
            f"- **# routes scored**: {len(scores)}"
        )

    st.markdown("---")
    st.caption(
        f"Schema-aligned with: {', '.join(dpp['dpp_meta']['regulatory_alignment'])}.  "
        f"Coverage: **{prov['coverage_pct'] * 100:.1f} %**  ·  "
        f"Schema validated: **{prov['schema_validated']}**"
    )


def render(
    selection,                # frontend.components.sidebar.Selection
    cell_df: pd.DataFrame,
    cycle_n: int,
    soh_out: dict,            # from soh_grade.render
    rul_out: dict,            # from rul.render
    mcdm_out: dict,           # from mcdm.render
) -> dict:
    with st.container(border=True):
        st.subheader("6 · Digital Product Passport")
        st.caption(
            "All upstream predictions assembled into a unified-schema DPP. "
            "The schema reconciles EU Regulation 2023/1542 Annex XIII, the GBA "
            "Battery Pass v1.2.0 data model, and India BWMR 2022 with its 2024 "
            "and 2025 amendments. Download the JSON below to inspect or pipe "
            "into your own validator."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_A **Digital Product Passport (DPP)** is the EU / India regulatory "
            "format — a structured JSON describing each battery's identity, "
            "health, carbon footprint, and recommended EoL route. The schema "
            "comes from the regulations themselves; the field values come from "
            "the predictions above._"
            "</span>",
            unsafe_allow_html=True,
        )

        cell_meta = selection.cell_meta
        cell_row = cell_df.iloc[(cell_df["cycle"] - cycle_n).abs().idxmin()]
        v_min = float(cell_df["v_min"].min()) if cell_df["v_min"].notna().any() else 2.0
        v_max = float(cell_df["v_max"].max()) if cell_df["v_max"].notna().any() else 4.2

        soh_for_dpp = soh_out["specialist_pred"]
        rul_for_dpp = rul_out["rul_xgb"]

        dpp = build_dpp(
            battery_id=cell_meta["battery_id"],
            chemistry=cell_meta["chemistry"],
            form_factor=cell_meta.get("form_factor"),
            nominal_Ah=cell_meta["nominal_Ah"],
            voltage_min_V=v_min,
            voltage_max_V=v_max,
            cycles_completed=int(cycle_n),
            soh_percent=float(soh_for_dpp),
            rul_remaining_cycles=float(rul_for_dpp),
            estimation_method=(
                "Ensemble of two machine-learning models: a global XGBoost "
                "regressor for State of Health, refined by a chemistry-specific "
                "specialist routed by the cell's detected chemistry; remaining "
                "useful life estimated by a separate XGBoost regressor."
            ),
            estimation_confidence={
                "metric": "test RMSE on held-out cells",
                "value": 0.0243,
                "validation_set": "Held-out test partition (cells the model never saw during training)",
            },
            data_source=("simulated" if cell_meta["source"].startswith("SYN_IN_")
                         else "field"),
            grade=mcdm_out["grade"],
            recommended_route=mcdm_out["top_alternative"],
            route_score=mcdm_out["closeness"],
            mcdm_criteria=list(mcdm_out["weights"].keys()),
            mcdm_weights=mcdm_out["weights"],
            all_route_scores=mcdm_out["ranked"],
            second_life=bool(cell_meta.get("second_life", False)),
            data_sources=[
                f"unified.parquet ({cell_meta['source']})",
                "data/processed/mcdm_weights/fuzzy_bwm_input.csv",
            ],
            model_artifacts=[
                "models/xgboost_soh/xgboost_soh_audited.json",
                "models/per_chemistry/router_manifest.json",
                "models/xgboost_rul/xgboost_rul_audited_uncensored.json",
                "models/tcn_rul/best.pt",
                "models/isolation_forest/isolation_forest.pkl",
                "models/vae_anomaly/best.pt",
                "src.mcdm.topsis.topsis_rank",
            ],
        )
        ok, errors = validate_against_schema(dpp)
        dpp["provenance"]["schema_validated"] = ok

        col_status, col_dl = st.columns([3, 1])
        if ok:
            col_status.success(
                f"DPP validates against unified schema  ·  "
                f"coverage {dpp['provenance']['coverage_pct'] * 100:.1f} %",
                icon=None,
            )
        else:
            col_status.error(
                f"DPP failed schema validation ({len(errors)} errors)",
                icon=None,
            )
            with st.expander("Show validation errors", expanded=False):
                for err in errors[:20]:
                    st.code(err)
        json_bytes = json.dumps(dpp, indent=2, default=str).encode("utf-8")
        col_dl.download_button(
            "Download JSON",
            data=json_bytes,
            file_name=f"dpp_{cell_meta['battery_id'].replace('/', '_')}.json",
            mime="application/json",
            width="stretch",
        )

        tab_summary, tab_json, tab_coverage = st.tabs([
            "Summary", "Raw JSON", "Field coverage"
        ])
        with tab_summary:
            _render_summary_card(dpp)
        with tab_json:
            st.code(json.dumps(dpp, indent=2, default=str), language="json")
        with tab_coverage:
            cov_df = _category_coverage(dpp)
            st.plotly_chart(_coverage_bar(cov_df), width="stretch")
            st.caption(
                "Carbon and supply-chain blocks intentionally hold placeholders "
                "in v1 — they require facility-specific manufacturing data that "
                "is out of scope for the demo cell pool."
            )

    return {"dpp": dpp, "validated": ok, "n_errors": len(errors)}
