"""
EDA 99 — Assemble all findings_*.md fragments into a single
data/processed/eda/EDA_REPORT.md with a table of contents and cross-summary.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "data" / "processed" / "eda"

ORDER = [
    ("01_cycling_overview", "L1 cycling — overview & missingness"),
    ("02_cycling_distributions", "L1 cycling — feature distributions"),
    ("03_cycling_degradation", "L1 cycling — degradation patterns"),
    ("04_cycling_correlations_outliers", "L1 cycling — correlations & outliers"),
    ("05_splits_audit", "L1 cycling — train/val/test split audit"),
    ("06_synthetic_indian", "L2 synthetic Indian-context cells"),
    ("07_regulatory_market", "L3+L4 regulatory + market data"),
    ("08_mcdm_weights", "L5 MCDM literature weights"),
]


def main():
    parts = []
    parts.append("# EDA Report — EV Battery EoL Routing Framework\n")
    parts.append(f"_Generated: {date.today().isoformat()} · "
                 f"Companion to [DATASET_IMPLEMENTATION_PLAN.md](../../../DATASET_IMPLEMENTATION_PLAN.md)._\n")
    parts.append(
        "This document consolidates exploratory data analysis across all 5 dataset layers. "
        "Each section corresponds to a self-contained `scripts/eda/<NN>_<topic>.py` script that "
        "regenerates its findings file and figures.\n"
    )

    # Table of contents
    parts.append("## Table of contents\n")
    for slug, title in ORDER:
        parts.append(f"- [{title}](#{slug.replace('_', '-')})")
    parts.append("")

    # Top-line summary
    parts.append("---\n## Top-line takeaways (executive summary)\n")
    parts.append("")
    parts.append("**Data quality**")
    parts.append("- Cycling corpus: 2.20 M rows · 1,521 batteries · 34 sources · 7 chemistries.")
    parts.append("- Critical label-quality issue: **6 rows had NaN `soh`** which crashed XGBoost full-data training; "
                 "patched in [src/data/training_data.py](../../../src/data/training_data.py) by dropping NaN/inf labels.")
    parts.append("- **`ir_ohm` is 93% missing** and temperature features are 87% missing — should be excluded from "
                 "the XGBoost feature set or imputed per-source.")
    parts.append("")
    parts.append("**Distributional findings**")
    parts.append("- **NMC dominates** the corpus by chemistry (75% of rows) and SoH mean is 0.35 — "
                 "the corpus is heavily late-life. LFP/Zn-ion test performance must be reported separately.")
    parts.append("- **`BL_ISU_ILCC` dominates** by source (54% of rows). Battery-level splits mitigate but "
                 "feature scaling is still ISU-biased.")
    parts.append("- **Coulombic efficiency outliers** are sparse (<1%) — useful labeled positives for the anomaly detectors.")
    parts.append("")
    parts.append("**Modeling implications**")
    parts.append("- `capacity_Ah` is the single strongest SoH feature (ρ ≈ 0.95+) — XGBoost will lean on it heavily; "
                 "the LSTM brings temporal information that capacity alone can't capture.")
    parts.append("- **Knee points** for ~80% of detected cells fall after 50% of cycle life — confirms our 30-cycle "
                 "LSTM context window is appropriate (sees pre-knee regime).")
    parts.append("- **Train/val/test splits are anti-leakage clean** (zero battery-ID overlap). Chemistry & SoH "
                 "distributions match within ~5 pp.")
    parts.append("")
    parts.append("**Regulatory landscape**")
    parts.append("- BWMR EV recovery target ramps 70 → 80 → 90% over 2024–27. Recycled-content target lags 3 years (2027–28 onset, 5 → 20% by 2030–31).")
    parts.append("- CPCB national procurement averages **only 58% of target** — even mature streams like Lead are below quota. "
                 "Validates literature-derived high weight on Compliance + EPR Return.")
    parts.append("- EU Annex XIII mandates 31 fields across 2 access tiers; our unified DPP schema reconciles "
                 "EU + GBA + BWMR (Stage 11 done).")
    parts.append("")
    parts.append("**MCDM input quality**")
    parts.append("- Strict-gate evidence (≥3 papers) for Compliance (6), EPR Return (4), Carbon (3). "
                 "SoH (0), Safety (1), Value (1) are structurally thin — fall back to 5-scenario sensitivity per §7.4.")
    parts.append("")
    parts.append("---\n")

    # Append each fragment
    for slug, _ in ORDER:
        f = EDA_DIR / f"findings_{slug}.md"
        if f.exists():
            parts.append(f"<a id=\"{slug.replace('_', '-')}\"></a>\n")
            parts.append(f.read_text())
            parts.append("\n---\n")

    out = EDA_DIR / "EDA_REPORT.md"
    out.write_text("\n".join(parts))
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
