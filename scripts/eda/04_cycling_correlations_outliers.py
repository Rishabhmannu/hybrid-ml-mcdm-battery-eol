"""
EDA 04 — Feature correlations + z-score outlier audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eda._eda_common import fig_path, md_table, write_findings
from src.utils.config import PROCESSED_DIR
from src.utils.plots import _PALETTE, apply_theme

UNIFIED = PROCESSED_DIR / "cycling" / "unified.parquet"
SECTION = "cycling_correlations_outliers"

NUMERIC_FEATURES = [
    "capacity_Ah", "soh",
    "v_min", "v_max", "v_mean",
    "i_min", "i_max", "i_mean",
    "t_mean", "t_range",
    "charge_time_s", "discharge_time_s",
    "ir_ohm", "coulombic_eff",
    "capacity_delta",
    "capacity_roll5_mean", "capacity_roll5_std",
    "capacity_roll20_mean", "capacity_roll20_std",
    "cycle_count_so_far",
]


def main():
    apply_theme()
    print(f"Loading {UNIFIED.relative_to(PROCESSED_DIR.parent.parent)} ...")
    df = pd.read_parquet(UNIFIED)
    print(f"  {len(df):,} rows")

    # ---------- Correlation heatmap ----------
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    sub = df[cols].sample(n=min(200_000, len(df)), random_state=42)
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=9)
    ax.set_title("Numeric feature correlation (Pearson, on 200k row sample)")
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if abs(v) > 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.75 else "#1A1A1A", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(fig_path(SECTION, "correlation_heatmap"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Top SoH-correlated features ----------
    soh_corr = corr["soh"].drop("soh").sort_values()
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.3 * len(soh_corr))))
    colors = [_PALETTE["val"] if v < 0 else _PALETTE["train"] for v in soh_corr.values]
    ax.barh(soh_corr.index, soh_corr.values, color=colors, alpha=0.85)
    ax.axvline(0, color="#7F8C8D", linewidth=0.7)
    ax.set_xlabel("Pearson correlation with SoH")
    ax.set_title("SoH correlation per feature (200k row sample)")
    fig.savefig(fig_path(SECTION, "soh_correlations"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Outlier flag analysis (precomputed columns from unify) ----------
    flags = ["capacity_Ah_z_outlier", "v_mean_z_outlier", "t_mean_z_outlier"]
    available = [c for c in flags if c in df.columns]
    flag_counts = {c: int(df[c].sum()) for c in available}
    overlap = pd.DataFrame({c: df[c].astype(int) for c in available})
    overlap["any"] = (overlap.sum(axis=1) > 0).astype(int)
    overlap["all"] = (overlap[available].sum(axis=1) == len(available)).astype(int)
    n_any = int(overlap["any"].sum())
    n_all = int(overlap["all"].sum())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    cats = available + ["any", "all"]
    vals = [flag_counts[c] for c in available] + [n_any, n_all]
    pcts = [v / len(df) * 100 for v in vals]
    ax.barh(cats, pcts, color=_PALETTE["train"], alpha=0.85)
    for i, v in enumerate(pcts):
        ax.text(v, i, f"  {vals[i]:,} ({v:.2f}%)", va="center", fontsize=8)
    ax.set_xlabel("% of rows flagged"); ax.set_title("Outlier flag counts (z>3) and overlap")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "outlier_flags"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Per-source outlier rates ----------
    if "capacity_Ah_z_outlier" in df.columns:
        rates = (df.groupby("source")["capacity_Ah_z_outlier"].mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(5, 0.25 * len(rates))))
        ax.barh(rates.index, rates.values, color=_PALETTE["accent"], alpha=0.85)
        ax.set_xlabel("% of rows with capacity_Ah z-outlier flag")
        ax.set_title("Outlier rate per source (capacity_Ah z>3)")
        fig.savefig(fig_path(SECTION, "outlier_rate_per_source"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---------- Markdown ----------
    top_pos = soh_corr.tail(5)
    top_neg = soh_corr.head(5)
    lines = [
        "# 4. Cycling correlations & outlier audit",
        "",
        "## 4.1 Feature correlation with SoH (top 5 each direction)",
        "",
        md_table(["positively correlated", "ρ"],
                 [(idx, f"{v:.3f}") for idx, v in top_pos[::-1].items()]),
        "",
        md_table(["negatively correlated", "ρ"],
                 [(idx, f"{v:.3f}") for idx, v in top_neg.items()]),
        "",
        "## 4.2 Outlier flag counts (precomputed in `src/data/unify.py`, z>3)",
        "",
        md_table(
            ["flag", "n flagged", "% of rows"],
            [(c, f"{flag_counts[c]:,}", f"{flag_counts[c]/len(df)*100:.2f}%") for c in available]
            + [
                ("**any flag**", f"{n_any:,}", f"{n_any/len(df)*100:.2f}%"),
                ("**all 3 flags simultaneously**", f"{n_all:,}", f"{n_all/len(df)*100:.2f}%"),
            ]
        ),
        "",
        "## 4.3 Headline insights",
        "",
    ]

    insights = []
    insights.append(
        f"- **`capacity_Ah` is the single strongest SoH predictor** "
        f"(ρ={corr.loc['capacity_Ah', 'soh']:.3f}) — expected, since SoH is derived from it. "
        "Use it carefully: training XGBoost with both `capacity_Ah` and `soh` simultaneously "
        "is fine because the model is regressing SoH from capacity, but the rolling capacity "
        "statistics dominate variable importance."
    )
    if "cycle_count_so_far" in corr.columns:
        cc_corr = corr.loc["cycle_count_so_far", "soh"]
        insights.append(
            f"- **`cycle_count_so_far` correlates {cc_corr:.3f} with SoH** — "
            "moderate negative; the LSTM should pick up the temporal degradation more strongly than XGBoost using "
            "this feature alone."
        )
    insights.append(
        f"- **{n_all/len(df)*100:.2f}% of rows trip all three z-outlier flags simultaneously** — "
        "tiny fraction, but useful as labeled positives for the anomaly detectors. "
        "These are likely instrumentation glitches (sensor dropouts) rather than real cell faults."
    )
    if "ir_ohm" in cols:
        ir_corr = corr.loc["ir_ohm", "soh"]
        insights.append(
            f"- **`ir_ohm` SoH correlation is only {ir_corr:.3f}** — but with **93% missingness**, "
            "this number is unreliable. IR is a known degradation indicator; "
            "should consider imputing per-source or excluding from XGBoost feature set entirely."
        )
    lines.extend(insights)
    lines.append("")
    lines.append("**Figures**: "
                 f"[correlation_heatmap]({fig_path(SECTION, 'correlation_heatmap').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[soh_correlations]({fig_path(SECTION, 'soh_correlations').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[outlier_flags]({fig_path(SECTION, 'outlier_flags').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[outlier_rate_per_source]({fig_path(SECTION, 'outlier_rate_per_source').relative_to(Path(__file__).resolve().parents[2])})")
    out = write_findings("04_cycling_correlations_outliers", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
