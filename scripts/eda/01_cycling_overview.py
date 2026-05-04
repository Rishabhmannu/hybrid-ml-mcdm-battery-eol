"""
EDA 01 — Cycling overview, schema audit, missingness.

Outputs:
- results/figures/eda/cycling_overview/  (4 figures)
- data/processed/eda/findings_01_cycling_overview.md
- data/processed/eda/cycling_schema_audit.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eda._eda_common import EDA_DIR, fig_path, md_table, write_findings
from src.utils.config import PROCESSED_DIR
from src.utils.plots import _PALETTE, apply_theme

UNIFIED = PROCESSED_DIR / "cycling" / "unified.parquet"

SECTION = "cycling_overview"


def main():
    apply_theme()
    print(f"Loading {UNIFIED.relative_to(PROCESSED_DIR.parent.parent)} ...")
    df = pd.read_parquet(UNIFIED)
    print(f"  {len(df):,} rows · {df['battery_id'].nunique():,} batteries · "
          f"{df['source'].nunique()} sources · {df['chemistry'].nunique()} chemistries")

    lines = [
        "# 1. L1 Cycling overview & missingness audit",
        "",
        f"_Source: `data/processed/cycling/unified.parquet` ({UNIFIED.stat().st_size/1e6:.1f} MB)_",
        "",
        f"- Rows: **{len(df):,}**",
        f"- Unique batteries: **{df['battery_id'].nunique():,}**",
        f"- Sources: **{df['source'].nunique()}**",
        f"- Chemistries: **{df['chemistry'].nunique()}**",
        f"- Form factors: **{df['form_factor'].nunique()}**",
        f"- Cycle range (max per battery): **{int(df.groupby('battery_id')['cycle'].max().min())}–{int(df.groupby('battery_id')['cycle'].max().max())}**",
        "",
    ]

    # ------------------------------------------------------------------
    # Schema + missingness audit (CSV + figure)
    # ------------------------------------------------------------------
    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean() * 100).round(3).values,
        "n_unique": [df[c].nunique() for c in df.columns],
    })
    audit.to_csv(EDA_DIR / "cycling_schema_audit.csv", index=False)
    print(f"  schema → {EDA_DIR.relative_to(PROCESSED_DIR.parent.parent)}/cycling_schema_audit.csv")

    high_missing = audit.sort_values("pct_missing", ascending=False).head(10)
    lines.append("## 1.1 Top columns by missing %")
    lines.append("")
    lines.append(md_table(
        ["column", "dtype", "missing %", "n_missing"],
        [(r["column"], r["dtype"], f"{r['pct_missing']:.2f}", f"{r['n_missing']:,}")
         for _, r in high_missing.iterrows()],
    ))
    lines.append("")

    # Missingness bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(audit))))
    sub = audit[audit["pct_missing"] > 0].sort_values("pct_missing", ascending=True)
    if len(sub) > 0:
        ax.barh(sub["column"], sub["pct_missing"], color=_PALETTE["train"], alpha=0.9)
        ax.set_xlabel("Missing %")
        ax.set_title("Per-column missingness (>0 only)")
    else:
        ax.text(0.5, 0.5, "No missing values across any column",
                transform=ax.transAxes, ha="center")
    fig.savefig(fig_path(SECTION, "missingness_bar"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Critical SoH/RUL label NaN check
    # ------------------------------------------------------------------
    soh_nan = df["soh"].isna().sum()
    soh_inf = (~np.isfinite(df["soh"])).sum() - soh_nan
    rows_with_zero_nominal = (df["nominal_Ah"] == 0).sum()

    lines.append("## 1.2 Critical label-quality flags")
    lines.append("")
    lines.append(md_table(
        ["check", "count", "pct of rows"],
        [
            ("`soh` NaN", f"{soh_nan:,}", f"{soh_nan/len(df)*100:.3f}%"),
            ("`soh` inf (after dropna)", f"{soh_inf:,}", f"{soh_inf/len(df)*100:.3f}%"),
            ("`nominal_Ah == 0`", f"{rows_with_zero_nominal:,}", f"{rows_with_zero_nominal/len(df)*100:.3f}%"),
        ],
    ))
    lines.append("")
    if soh_nan > 0 or soh_inf > 0 or rows_with_zero_nominal > 0:
        lines.append("> **⚠ Action required:** these rows must be dropped before XGBoost / LSTM "
                     "training (XGBoost rejects NaN labels). Patched in "
                     "[src/data/training_data.py::_build_feature_frame](../../../src/data/training_data.py).")
    lines.append("")

    # Per-source breakdown of missing SoH
    src_missing = (df.assign(soh_nan=df["soh"].isna())
                   .groupby("source", as_index=False)
                   .agg(n_rows=("soh", "size"),
                        soh_nan_pct=("soh_nan", lambda s: s.mean() * 100),
                        n_zero_nominal=("nominal_Ah", lambda s: (s == 0).sum()))
                   .sort_values("soh_nan_pct", ascending=False))
    if (src_missing["soh_nan_pct"] > 0).any():
        lines.append("### Per-source SoH-NaN concentration (top 10)")
        lines.append("")
        lines.append(md_table(
            ["source", "rows", "soh NaN %", "rows with nominal=0"],
            [(r["source"], f"{r['n_rows']:,}", f"{r['soh_nan_pct']:.2f}%",
              f"{r['n_zero_nominal']:,}")
             for _, r in src_missing.head(10).iterrows()
             if r["soh_nan_pct"] > 0 or r["n_zero_nominal"] > 0],
        ))
        lines.append("")

    # ------------------------------------------------------------------
    # Source × chemistry matrix
    # ------------------------------------------------------------------
    sx = (df.groupby(["source", "chemistry"])["battery_id"].nunique()
          .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.7 * sx.shape[1]),
                                     max(4, 0.28 * sx.shape[0])))
    im = ax.imshow(sx.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(sx.shape[1])); ax.set_xticklabels(sx.columns, rotation=45, ha="right")
    ax.set_yticks(range(sx.shape[0])); ax.set_yticklabels(sx.index, fontsize=8)
    ax.set_title("Source × Chemistry — number of batteries")
    for i in range(sx.shape[0]):
        for j in range(sx.shape[1]):
            v = sx.values[i, j]
            if v > 0:
                color = "white" if v > sx.values.max() * 0.55 else "#1A1A1A"
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.savefig(fig_path(SECTION, "source_x_chemistry"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-source counts (rows + batteries)
    # ------------------------------------------------------------------
    by_src = (df.groupby("source")
              .agg(n_rows=("battery_id", "size"),
                   n_batteries=("battery_id", "nunique"),
                   median_cycles=("cycle", lambda s: s.groupby(df.loc[s.index, "battery_id"]).max().median()))
              .sort_values("n_rows", ascending=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.25 * len(by_src))))
    axes[0].barh(by_src.index, by_src["n_rows"], color=_PALETTE["train"], alpha=0.9)
    axes[0].set_xlabel("Rows (cycle records)"); axes[0].set_title("Rows per source")
    axes[0].set_xscale("log")
    axes[1].barh(by_src.index, by_src["n_batteries"], color=_PALETTE["accent"], alpha=0.9)
    axes[1].set_xlabel("Unique batteries"); axes[1].set_title("Batteries per source")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "per_source_counts"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines.append("## 1.3 Per-source dataset shape (top 10)")
    lines.append("")
    top = by_src.sort_values("n_rows", ascending=False).head(10)
    lines.append(md_table(
        ["source", "rows", "batteries", "median cycles/battery"],
        [(idx, f"{int(r['n_rows']):,}", int(r["n_batteries"]),
          f"{r['median_cycles']:.0f}" if pd.notna(r["median_cycles"]) else "—")
         for idx, r in top.iterrows()],
    ))
    lines.append("")

    # ------------------------------------------------------------------
    # Cycle counts by source — boxplot
    # ------------------------------------------------------------------
    cycles_per_battery = df.groupby(["source", "battery_id"])["cycle"].max().reset_index()
    sources_sorted = (cycles_per_battery.groupby("source")["cycle"].median()
                      .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(sources_sorted))))
    box_data = [cycles_per_battery[cycles_per_battery["source"] == s]["cycle"].values
                for s in sources_sorted]
    bp = ax.boxplot(box_data, vert=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=_PALETTE["val"], linewidth=1.5))
    for box in bp["boxes"]:
        box.set(facecolor=_PALETTE["train"], alpha=0.6)
    ax.set_yticklabels(sources_sorted, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Cycles per battery (log)")
    ax.set_title("Cycle-count distribution per source (log scale)")
    fig.savefig(fig_path(SECTION, "cycle_count_per_source"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Chemistry summary
    # ------------------------------------------------------------------
    chem_summary = (df.groupby("chemistry")
                    .agg(n_rows=("battery_id", "size"),
                         n_batteries=("battery_id", "nunique"),
                         soh_mean=("soh", "mean"),
                         soh_min=("soh", "min"),
                         soh_max=("soh", "max"))
                    .sort_values("n_rows", ascending=False))
    lines.append("## 1.4 Per-chemistry coverage")
    lines.append("")
    lines.append(md_table(
        ["chemistry", "rows", "batteries", "SoH mean", "SoH min", "SoH max"],
        [(idx, f"{int(r['n_rows']):,}", int(r["n_batteries"]),
          f"{r['soh_mean']:.3f}" if pd.notna(r["soh_mean"]) else "—",
          f"{r['soh_min']:.3f}" if pd.notna(r["soh_min"]) else "—",
          f"{r['soh_max']:.3f}" if pd.notna(r["soh_max"]) else "—")
         for idx, r in chem_summary.iterrows()],
    ))
    lines.append("")

    # ------------------------------------------------------------------
    # Headline insights
    # ------------------------------------------------------------------
    lines.append("## 1.5 Headline insights")
    lines.append("")
    insights = []
    if soh_nan > 0:
        insights.append(f"- **{soh_nan:,} rows have NaN `soh`** ({soh_nan/len(df)*100:.2f}%) — "
                        f"these MUST be dropped before training (caused the 2026-04-28 XGBoost crash).")
    if rows_with_zero_nominal > 0:
        insights.append(f"- **{rows_with_zero_nominal:,} rows have `nominal_Ah=0`** — "
                        f"likely Zn-ion coin cells where capacity is reported in mAh; "
                        f"check `BL_ZN-coin` source for unit mismatches.")
    dom_src = by_src["n_rows"].idxmax()
    dom_pct = by_src["n_rows"].max() / by_src["n_rows"].sum() * 100
    insights.append(f"- **`{dom_src}` dominates the corpus** "
                    f"({dom_pct:.1f}% of all rows) — train/val/test splits stratified at "
                    f"the battery level mitigate, but feature scalers will still be biased "
                    f"toward this distribution.")
    nmc_pct = chem_summary.loc["NMC", "n_rows"] / chem_summary["n_rows"].sum() * 100 \
              if "NMC" in chem_summary.index else 0
    insights.append(f"- **NMC dominates by chemistry** ({nmc_pct:.1f}% of rows) — "
                    "expect models to be NMC-biased; LFP/Zn-ion test performance should be reported separately.")
    lines.extend(insights)
    lines.append("")
    lines.append("**Figures**: "
                 f"[missingness_bar]({fig_path(SECTION, 'missingness_bar').relative_to(PROJECT_ROOT)}) · "
                 f"[source_x_chemistry]({fig_path(SECTION, 'source_x_chemistry').relative_to(PROJECT_ROOT)}) · "
                 f"[per_source_counts]({fig_path(SECTION, 'per_source_counts').relative_to(PROJECT_ROOT)}) · "
                 f"[cycle_count_per_source]({fig_path(SECTION, 'cycle_count_per_source').relative_to(PROJECT_ROOT)})")
    lines.append("")

    out = write_findings("01_cycling_overview", lines)
    print(f"  findings → {out.relative_to(PROCESSED_DIR.parent.parent)}")
    print("Done.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    main()
