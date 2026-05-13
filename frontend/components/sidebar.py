"""Sidebar: chemistry filter → grade filter → cell picker → cycle slider → regime.

A guided sidebar — every widget carries a short hover tooltip explaining the
domain term, and each demo cell has its own "what to look for" story so the
user knows exactly what the panels below should show.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = FRONTEND_ROOT / "data" / "demo_cells_manifest.json"
STORIES_PATH = FRONTEND_ROOT / "data" / "cell_stories.json"

# Cell shown first when the user lands on the demo with no active filter.
# Picked because it's a clean grade-A LFP with 1481 cycles that passes both
# anomaly gates — a strong first impression of the full pipeline.
FEATURED_CELL_ID = "BL_HUST_HUST_2-8"

REGIMES = [
    "Literature Mean",
    "BWMR-Heavy",
    "EU-Heavy",
    "Technical-Heavy",
    "Equal Weights",
]

REGIME_BLURBS = {
    "Literature Mean":  "Defuzzified Fuzzy-BWM weights aggregated across 12+ MCDM papers.",
    "BWMR-Heavy":       "India BWMR 2022: Compliance (0.40) + EPR Return (0.25) dominant.",
    "EU-Heavy":         "EU Regulation 2023/1542: Carbon (0.35) + Compliance (0.30) dominant.",
    "Technical-Heavy":  "Engineering view: SoH (0.30) + Value (0.25) dominant.",
    "Equal Weights":    "All six criteria weighted 1/6 — sanity baseline.",
}

# Tooltip text shown on hover next to each sidebar widget label.
TOOLTIPS = {
    "chemistry": (
        "Electrode material that determines aging behaviour and end-of-life "
        "value. NMC is automotive-grade, LFP is common in e-bikes / buses, "
        "NCA in early Teslas, LCO in old laptops, Zn-ion and Na-ion are "
        "post-lithium alternatives for grid / stationary storage."
    ),
    "grade": (
        "Quality grade derived from State-of-Health: A > 80 % SoH (second-life "
        "ready), B 60–80 % (refurbishable), C 40–60 % (limited reuse / repair), "
        "D ≤ 40 % (recycle only)."
    ),
    "cell": (
        "Individual lithium-ion (or alternative-chemistry) cells from public lab "
        "datasets — BatteryLife, NASA-PCOE, CALCE, Stanford — plus 2 synthetic "
        "Indian-context cells simulated with PyBaMM and BLAST-Lite."
    ),
    "cycle": (
        "Each cycle is one charge-discharge event. Batteries lose ~0.01–0.05 % "
        "SoH per cycle on average. Move the slider to predict at any earlier "
        "point in the cell's life — the panels below will refresh."
    ),
    "regime": (
        "Different jurisdictions weight EoL criteria differently. Flipping the "
        "regime changes the criterion weights and can change the recommended "
        "route — that flip is RQ2 of the paper."
    ),
}


@dataclass(frozen=True)
class Selection:
    battery_id: str
    cycle_n: int
    regime: str
    cell_meta: dict


@st.cache_data
def load_manifest() -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    cells = m["cells"]
    # Promote the featured cell to position 0 so it's the cold-open default.
    idx = next((i for i, c in enumerate(cells) if c["battery_id"] == FEATURED_CELL_ID), None)
    if idx is not None and idx != 0:
        cells.insert(0, cells.pop(idx))
        m["cells"] = cells
    return m


@st.cache_data
def load_stories() -> dict:
    return json.loads(STORIES_PATH.read_text()) if STORIES_PATH.exists() else {}


def _filter_cells(cells: list[dict], chemistry: str, grade: str) -> list[dict]:
    out = cells
    if chemistry != "Any":
        out = [c for c in out if c["chemistry"] == chemistry]
    if grade != "Any":
        out = [c for c in out if c["grade_at_end"] == grade]
    return out


def render_sidebar() -> Selection | None:
    """Render the sidebar, return the user's selection (or None if nothing matches)."""
    manifest = load_manifest()
    stories = load_stories()
    cells = manifest["cells"]

    with st.sidebar:
        st.markdown("### Select a demo battery")
        chemistries = ["Any"] + sorted({c["chemistry"] for c in cells})
        grades = ["Any", "A", "B", "C", "D"]
        chem_choice = st.selectbox(
            "Chemistry", chemistries,
            key="filter_chemistry", help=TOOLTIPS["chemistry"],
        )
        grade_choice = st.selectbox(
            "Grade at end of trajectory", grades,
            key="filter_grade", help=TOOLTIPS["grade"],
        )

        filtered = _filter_cells(cells, chem_choice, grade_choice)
        if not filtered:
            st.warning("No demo cells match this filter combination.")
            return None

        def _label(c: dict) -> str:
            tag = "synthetic" if c["source"].startswith("SYN_IN_") else "real"
            return (
                f"{c['battery_id']}  ·  {c['chemistry']}/{c['grade_at_end']}  "
                f"·  {c['n_cycles']} cyc  ·  {tag}"
            )

        labels = [_label(c) for c in filtered]
        idx = st.selectbox(
            f"Cell ({len(filtered)} match)",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="cell_picker", help=TOOLTIPS["cell"],
        )
        cell_meta = filtered[idx]

        with st.expander("Cell metadata", expanded=True):
            st.markdown(
                f"**Source**: {cell_meta['source']}  \n"
                f"**Form factor**: {cell_meta.get('form_factor') or 'n/a'}  \n"
                f"**Nominal capacity**: {cell_meta['nominal_Ah']} Ah  \n"
                f"**Cycles observed**: {cell_meta['n_cycles']}  \n"
                f"**SoH first → last**: "
                f"{cell_meta['soh_first']:.3f} → {cell_meta['soh_last']:.3f}  \n"
                f"**Right-censored**: {'yes' if cell_meta['censored'] else 'no'}  \n"
                f"**Second-life**: {'yes' if cell_meta['second_life'] else 'no'}"
            )

        story = stories.get(cell_meta["battery_id"])
        if story:
            st.markdown(
                f"<div style='background:#FFFBE6;border-left:3px solid #C7A92E;"
                f"padding:0.6rem 0.8rem;margin:0.4rem 0;font-size:0.9rem;'>"
                f"<b>What to look for</b> — {story['headline']}<br>"
                f"<span style='color:#444'>{story['walkthrough']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        cycle_n = st.slider(
            "Predict at cycle",
            min_value=1,
            max_value=cell_meta["cycle_max"],
            value=cell_meta["cycle_max"],
            help=TOOLTIPS["cycle"],
            key="cycle_slider",
        )

        st.divider()
        st.markdown("### Regulatory regime (RQ2)")
        regime = st.radio(
            "Weight scenario",
            REGIMES,
            index=0,
            key="regime_choice",
            label_visibility="collapsed",
            help=TOOLTIPS["regime"],
        )
        st.caption(REGIME_BLURBS[regime])

        st.divider()
        st.caption(
            f"Demo cell pool: **{manifest['n_cells']} cells** "
            f"({manifest['n_real']} real · {manifest['n_synthetic']} synthetic)"
        )

    return Selection(
        battery_id=cell_meta["battery_id"],
        cycle_n=int(cycle_n),
        regime=regime,
        cell_meta=cell_meta,
    )
