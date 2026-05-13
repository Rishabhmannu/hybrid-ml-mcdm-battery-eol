"""Streamlit entry point — Hybrid ML-MCDM EV Battery EoL Routing demo.

Run locally:
    streamlit run frontend/app.py

Pipeline rendered top-to-bottom for the selected cell:
    Cell trajectory → Anomaly → SoH + Grade → RUL → MCDM routing → DPP JSON
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import anomaly, dpp, mcdm, rul, soh_grade, trajectory  # noqa: E402
from frontend.components.sidebar import load_stories, render_sidebar  # noqa: E402

DEMO_CELLS_PATH = FRONTEND_ROOT / "data" / "demo_cells.parquet"


@st.cache_data
def load_demo_cells() -> pd.DataFrame:
    return pd.read_parquet(DEMO_CELLS_PATH)


st.set_page_config(
    page_title="EV Battery EoL Routing — ML-MCDM Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_header() -> None:
    st.title("Hybrid ML-MCDM Framework for EV Battery End-of-Life Routing")
    st.markdown(
        "An interactive walkthrough of the full pipeline — pick a demo cell in "
        "the sidebar, then watch it move through anomaly gating, state-of-health "
        "estimation, remaining-useful-life prediction, regulatory-aware route "
        "selection, and Digital Product Passport emission."
    )
    cols = st.columns(2)
    cols[0].markdown("**Code**: GitHub repo (link TBD)")
    cols[1].markdown("**Paper**: preprint (link TBD)")


def _filename_safe(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def _build_pdf_lazy(selection, cell_df, anomaly_out, soh_out, rul_out,
                    mcdm_out, dpp_out) -> bytes:
    """Build PDF bytes on demand. Imports kept local so cold-start stays cheap."""
    from frontend.components.mcdm import _load_scenarios
    from frontend.components.soh_grade import _load_router
    from frontend.lib.report_pdf import build_cell_report

    _, _, chem_stats = _load_router()
    scenarios = _load_scenarios()
    story = load_stories().get(selection.battery_id)
    return build_cell_report(
        selection=selection,
        cell_df=cell_df,
        anomaly_out=anomaly_out,
        soh_out=soh_out,
        rul_out=rul_out,
        mcdm_out=mcdm_out,
        dpp_out=dpp_out,
        cell_story=story,
        chem_stats=chem_stats,
        scenarios=scenarios,
    )


def main() -> None:
    render_header()

    selection = render_sidebar()
    if selection is None:
        st.stop()

    st.divider()
    info_col, pdf_col, json_col = st.columns([2, 1, 1])
    info_col.markdown(
        f"#### Selected: `{selection.battery_id}`  ·  "
        f"cycle **{selection.cycle_n}** of {selection.cell_meta['cycle_max']}  ·  "
        f"regime: **{selection.regime}**"
    )
    pdf_slot = pdf_col.empty()
    json_slot = json_col.empty()

    demo = load_demo_cells()
    cell_df = (demo[demo["battery_id"] == selection.battery_id]
               .sort_values("cycle").reset_index(drop=True))

    trajectory.render(cell_df, selection.cycle_n)
    anomaly_out = anomaly.render(cell_df, selection.cycle_n)
    soh_out = soh_grade.render(cell_df, selection.cycle_n)
    rul_out = rul.render(cell_df, selection.cycle_n)
    mcdm_out = mcdm.render(grade=soh_out["grade"], regime=selection.regime)
    dpp_out = dpp.render(selection, cell_df, selection.cycle_n,
                         soh_out, rul_out, mcdm_out)

    # ---- JSON download (instant, always-on) ----
    json_bytes = json.dumps(dpp_out["dpp"], indent=2, default=str).encode("utf-8")
    json_slot.download_button(
        "Download DPP (JSON)",
        data=json_bytes,
        file_name=f"dpp_{_filename_safe(selection.battery_id)}.json",
        mime="application/json",
        width="stretch",
        help="The machine-readable Digital Product Passport for this cell.",
    )

    # ---- PDF report (lazy — built on click, cached per selection) ----
    pdf_key = (
        f"pdf__{selection.battery_id}__{selection.cycle_n}__{selection.regime}"
    )
    if pdf_key in st.session_state:
        date_tag = _dt.datetime.now().strftime("%Y%m%d")
        pdf_slot.download_button(
            "Download PDF report",
            data=st.session_state[pdf_key],
            file_name=f"cell_report_{_filename_safe(selection.battery_id)}_{date_tag}.pdf",
            mime="application/pdf",
            width="stretch",
            type="primary",
            help="The full human-readable cell report — 8-page PDF.",
        )
        pdf_col.caption("Report cached for this cell + cycle + regime.")
    else:
        clicked = pdf_slot.button(
            "Generate PDF report",
            width="stretch",
            help="Build a print-ready PDF of this cell's full report.",
        )
        pdf_col.caption(
            "⏱ ~25–30 s — renders all charts at print quality. "
            "Cached after first generation for this cell + cycle + regime."
        )
        if clicked:
            # Render the spinner inside the button's column so the user gets
            # visible feedback without having to scroll to the bottom of the page.
            with pdf_col.container():
                with st.spinner("Building PDF report — 25–30 s…"):
                    st.session_state[pdf_key] = _build_pdf_lazy(
                        selection, cell_df, anomaly_out, soh_out, rul_out,
                        mcdm_out, dpp_out,
                    )
            st.rerun()


if __name__ == "__main__":
    main()
