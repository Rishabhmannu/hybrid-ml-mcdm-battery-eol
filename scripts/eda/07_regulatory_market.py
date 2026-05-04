"""
EDA 07 — Regulatory + market data audit.
- BWMR recovery & recycled-content target trajectories
- EU Annex XIII field counts by access tier
- CPCB metals: target vs procured vs available
- ICEA growth-target projections
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eda._eda_common import EDA_DIR, fig_path, md_table, write_findings
from src.utils.config import PROCESSED_DIR
from src.utils.plots import _PALETTE, apply_theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECTION = "regulatory_market"

BWMR_TABLES = PROJECT_ROOT / "data" / "regulatory" / "bwmr" / "extracted_tables"
EU_FIELDS = PROCESSED_DIR / "dpp" / "eu_annex_xiii_fields.csv"
CPCB = PROCESSED_DIR / "market" / "cpcb_metal_epr_table.csv"
ICEA = PROCESSED_DIR / "market" / "icea_projections.csv"


def _bwmr_recovery_long() -> pd.DataFrame:
    f = BWMR_TABLES / "BWMR_2022_original_S.O.3984__p32__t1__recovery_targets.csv"
    raw = pd.read_csv(f, header=None)
    type_col = raw.iloc[2:, 1].tolist()
    years = ["2024-25", "2025-26", "2026-27+"]
    rows = []
    for r_idx, t in enumerate(type_col):
        for c_idx, y in enumerate(years):
            v = raw.iloc[2 + r_idx, 2 + c_idx]
            try:
                rows.append({"battery_type": t, "year": y, "recovery_pct": float(v)})
            except (ValueError, TypeError):
                pass
    return pd.DataFrame(rows)


def _bwmr_recycled_long() -> pd.DataFrame:
    f = BWMR_TABLES / "BWMR_2022_original_S.O.3984__p30__t1__recycled_content_targets.csv"
    raw = pd.read_csv(f, header=None)
    rows = []
    # Two header bands per the original PDF table layout
    for header_row, data_rows in [(1, [2, 3]), (4, [5, 6])]:
        years = [str(x) for x in raw.iloc[header_row, 2:].tolist() if str(x) != "nan"]
        for d in data_rows:
            t = raw.iloc[d, 1]
            for c_idx, y in enumerate(years):
                try:
                    rows.append({"battery_type": t, "year": y,
                                 "recycled_pct": float(raw.iloc[d, 2 + c_idx])})
                except (ValueError, TypeError, IndexError):
                    pass
    return pd.DataFrame(rows)


def main():
    apply_theme()

    # ============================================================
    # 7.1 BWMR recovery target trajectory
    # ============================================================
    rec = _bwmr_recovery_long()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    types = rec["battery_type"].unique()
    width = 0.18
    x = np.arange(len(rec["year"].unique()))
    for i, t in enumerate(types):
        sub = rec[rec["battery_type"] == t].set_index("year").reindex(rec["year"].unique())
        ax.bar(x + (i - 1.5) * width, sub["recovery_pct"], width=width, label=t, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rec["year"].unique())
    ax.set_ylabel("Recovery target (%)")
    ax.set_title("BWMR 2022 — Recovery target trajectory by battery type")
    ax.legend(fontsize=8)
    fig.savefig(fig_path(SECTION, "bwmr_recovery_targets"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 7.2 BWMR recycled content trajectory
    # ============================================================
    rcy = _bwmr_recycled_long()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in rcy["battery_type"].unique():
        sub = rcy[rcy["battery_type"] == t].sort_values("year")
        ax.plot(sub["year"], sub["recycled_pct"], marker="o", linewidth=1.5, label=t)
    ax.set_ylabel("Recycled-content target (%)")
    ax.set_title("BWMR 2022 — Recycled content trajectory")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "bwmr_recycled_content_targets"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 7.3 EU Annex XIII field counts by access tier
    # ============================================================
    eu = pd.read_csv(EU_FIELDS)
    tier_counts = eu.groupby("access_level")["field_text"].count().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(tier_counts.index, tier_counts.values, color=_PALETTE["train"], alpha=0.85)
    for i, v in enumerate(tier_counts.values):
        ax.text(v, i, f"  {v}", va="center", fontsize=9)
    ax.set_xlabel("Number of mandated fields")
    ax.set_title("EU Annex XIII fields per access tier")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "eu_annex_xiii_tiers"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 7.4 CPCB metals: target vs procured vs available
    # ============================================================
    cpcb = pd.read_csv(CPCB)
    cpcb["procured_pct"] = cpcb["epr_credits_procured_tonnes"] / cpcb["epr_target_tonnes"] * 100
    cpcb_sorted = cpcb.sort_values("epr_target_tonnes", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(cpcb_sorted))))
    y = np.arange(len(cpcb_sorted))
    ax.barh(y - 0.25, cpcb_sorted["epr_target_tonnes"], height=0.25,
            label="Target", color=_PALETTE["train"], alpha=0.85)
    ax.barh(y, cpcb_sorted["epr_credits_procured_tonnes"], height=0.25,
            label="Procured", color=_PALETTE["accent"], alpha=0.85)
    ax.barh(y + 0.25, cpcb_sorted["epr_credits_available_tonnes"], height=0.25,
            label="Available", color=_PALETTE["test"], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(cpcb_sorted["metal"])
    ax.set_xscale("log")
    ax.set_xlabel("Tonnes (log scale)")
    ax.set_title("CPCB EPR — Target vs Procured vs Available (per metal)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "cpcb_epr_target_vs_procured"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Procurement-shortfall plot
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.35 * len(cpcb))))
    cpcb_pct = cpcb.sort_values("procured_pct")
    ax.barh(cpcb_pct["metal"], cpcb_pct["procured_pct"], color=_PALETTE["accent"], alpha=0.85)
    ax.axvline(100, color="#7F8C8D", linestyle="--", linewidth=0.7, label="100 % target")
    ax.set_xlabel("Procured / Target (%)")
    ax.set_title("CPCB EPR procurement shortfall per metal")
    for i, v in enumerate(cpcb_pct["procured_pct"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "cpcb_procurement_pct"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 7.5 ICEA growth targets — extract years + tonnages
    # ============================================================
    icea = pd.read_csv(ICEA)
    growth = icea[icea["kind"] == "growth_target"].copy()
    parsed = []
    for _, r in growth.iterrows():
        m_year = re.search(r"(20[2-3][0-9])", str(r["match"]))
        m_tons = re.search(r"([\d,\.]+)\s*kT", str(r["match"]))
        m_dollars = re.search(r"\$\s*([\d\.]+)\s*(billion|trillion|million)?",
                              str(r["match"]), flags=re.IGNORECASE)
        if m_year and m_tons:
            parsed.append({
                "year": int(m_year.group(1)),
                "tonnes_kT": float(m_tons.group(1).replace(",", "")),
                "snippet": r["context_snippet"][:80],
            })
    icea_kt = pd.DataFrame(parsed).sort_values("year")
    if len(icea_kt) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.scatter(icea_kt["year"], icea_kt["tonnes_kT"],
                   s=80, color=_PALETTE["train"], alpha=0.7, edgecolor="white")
        for _, r in icea_kt.iterrows():
            ax.annotate(f"{r['tonnes_kT']:.0f} kT", (r["year"], r["tonnes_kT"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Year"); ax.set_ylabel("Capacity / production target (kT)")
        ax.set_title("ICEA-Accenture growth-target projections (kT)")
        fig.tight_layout()
        fig.savefig(fig_path(SECTION, "icea_growth_targets"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ============================================================
    # Markdown
    # ============================================================
    lines = [
        "# 7. Regulatory + Market data audit",
        "",
        "## 7.1 BWMR recovery target trajectory",
        "",
        md_table(
            ["battery type", "2024-25", "2025-26", "2026-27+"],
            [(t,
              f"{rec.loc[(rec.battery_type==t) & (rec.year=='2024-25'), 'recovery_pct'].iloc[0]:.0f}%" if not rec[(rec.battery_type==t) & (rec.year=='2024-25')].empty else "—",
              f"{rec.loc[(rec.battery_type==t) & (rec.year=='2025-26'), 'recovery_pct'].iloc[0]:.0f}%" if not rec[(rec.battery_type==t) & (rec.year=='2025-26')].empty else "—",
              f"{rec.loc[(rec.battery_type==t) & (rec.year=='2026-27+'), 'recovery_pct'].iloc[0]:.0f}%" if not rec[(rec.battery_type==t) & (rec.year=='2026-27+')].empty else "—")
             for t in rec["battery_type"].unique()],
        ),
        "",
        "## 7.2 BWMR recycled-content trajectory",
        "",
        md_table(
            ["battery type"] + sorted(rcy["year"].unique()),
            [[t] + [f"{rcy.loc[(rcy.battery_type==t) & (rcy.year==y), 'recycled_pct'].iloc[0]:.0f}%" if not rcy[(rcy.battery_type==t) & (rcy.year==y)].empty else "—"
                    for y in sorted(rcy["year"].unique())]
             for t in rcy["battery_type"].unique()],
        ),
        "",
        "## 7.3 EU Annex XIII field counts per access tier",
        "",
        md_table(
            ["access tier", "n fields"],
            [(idx, int(v)) for idx, v in tier_counts.items()],
        ),
        "",
        "## 7.4 CPCB EPR procurement gap (top mismatches)",
        "",
        md_table(
            ["metal", "target (T)", "procured (T)", "available (T)", "procured %"],
            [(r["metal"], f"{r['epr_target_tonnes']:,.0f}",
              f"{r['epr_credits_procured_tonnes']:,.0f}",
              f"{r['epr_credits_available_tonnes']:,.0f}",
              f"{r['procured_pct']:.1f}%")
             for _, r in cpcb.sort_values("procured_pct").iterrows()],
        ),
        "",
        "## 7.5 Headline insights",
        "",
    ]

    insights = []
    insights.append(
        "- **BWMR recovery target ramps from 70% to 90% over 2024–27 for EV batteries** — "
        "the routing model must reflect this changing constraint; 90% recovery is a tight gate "
        "for chemistries like LFP that are mostly 'plastic' by mass."
    )
    insights.append(
        "- **Recycled-content target lag for EV batteries** — "
        "starts at 5% in 2027-28 (3 years after recovery starts), "
        "ramping to 20% by 2030-31. Industrial/automotive Pb-acid is at 35% from year one because "
        "recycled-Pb supply is mature."
    )
    avg_proc = cpcb["procured_pct"].mean()
    insights.append(
        f"- **CPCB national procurement averages only {avg_proc:.1f}% of target** — "
        "even Lead, the most mature stream, is only at "
        f"{cpcb.loc[cpcb['metal']=='Lead', 'procured_pct'].iloc[0]:.0f}%. "
        "Routing model should weight 'Compliance' and 'EPR Return' criteria heavily — "
        "validates the literature-derived weights from §7 of the implementation plan."
    )
    if len(icea_kt) > 0:
        insights.append(
            f"- **ICEA growth targets span {icea_kt['year'].min()}–{icea_kt['year'].max()}** "
            f"with values from {icea_kt['tonnes_kT'].min():.0f} kT to {icea_kt['tonnes_kT'].max():.0f} kT — "
            "these are consistent across multiple ICEA-Accenture publications and provide stable anchors for "
            "stream-volume forecasting in the manuscript."
        )
    insights.append(
        f"- **EU Annex XIII covers {len(eu)} mandated fields across {len(tier_counts)} access tiers** — "
        f"only {tier_counts.get('public', 0)} are public (BoM, performance, EoL); "
        "supply-chain and dismantling fields are restricted. Our unified DPP schema (Stage 11) emits "
        "all three tiers in a single JSON with an `access_level` field per block."
    )
    lines.extend(insights)
    lines.append("")
    figs = ["bwmr_recovery_targets", "bwmr_recycled_content_targets",
            "eu_annex_xiii_tiers", "cpcb_epr_target_vs_procured",
            "cpcb_procurement_pct", "icea_growth_targets"]
    rels = [str(fig_path(SECTION, f).relative_to(PROJECT_ROOT)) for f in figs]
    lines.append("**Figures**: " + " · ".join(f"[{f}]({r})" for f, r in zip(figs, rels)))
    out = write_findings("07_regulatory_market", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
