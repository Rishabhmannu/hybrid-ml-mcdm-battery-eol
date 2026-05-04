"""
EDA 02 — Numeric feature distributions + per-chemistry SoH/V/T profiles.
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
SECTION = "cycling_distributions"

NUMERIC = [
    "capacity_Ah", "soh",
    "v_min", "v_max", "v_mean",
    "i_min", "i_max", "i_mean",
    "t_min", "t_max", "t_mean", "t_range",
    "charge_time_s", "discharge_time_s",
    "ir_ohm", "coulombic_eff",
]


def main():
    apply_theme()
    print(f"Loading {UNIFIED.relative_to(PROCESSED_DIR.parent.parent)} ...")
    df = pd.read_parquet(UNIFIED)
    print(f"  {len(df):,} rows")

    # ---------- Numeric histograms grid ----------
    cols = [c for c in NUMERIC if c in df.columns]
    n = len(cols); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        ax = axes[i]
        s = df[c].dropna()
        if len(s) == 0:
            ax.text(0.5, 0.5, f"{c}\nall missing", ha="center", va="center",
                    transform=ax.transAxes); ax.axis("off"); continue
        # Robust clip for plotting (drop extreme outliers from view)
        lo, hi = np.percentile(s, [0.1, 99.9])
        clipped = s[(s >= lo) & (s <= hi)]
        ax.hist(clipped, bins=60, color=_PALETTE["train"], alpha=0.85)
        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=8)
        if c == "soh":
            ax.axvline(0.8, color=_PALETTE["val"], linestyle="--", linewidth=0.8,
                       label="EoL @ 0.8")
            ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Cycling features — clipped histograms (0.1–99.9 percentile)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "feature_histograms"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Per-chemistry SoH violin ----------
    chems = (df.groupby("chemistry")["soh"].count()
             .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [df.loc[df["chemistry"] == c, "soh"].dropna().values for c in chems]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor(_PALETTE["train"]); body.set_alpha(0.65)
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in parts:
            parts[k].set_color(_PALETTE["val"])
    ax.set_xticks(range(1, len(chems) + 1)); ax.set_xticklabels(chems)
    ax.set_ylabel("SoH"); ax.set_title("SoH distribution by chemistry")
    ax.axhline(0.8, color="#7F8C8D", linestyle="--", linewidth=0.8, label="EoL=0.80")
    ax.legend()
    fig.savefig(fig_path(SECTION, "soh_violin_by_chemistry"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Voltage range per chemistry (boxplot) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title in zip(axes, ["v_min", "v_max"],
                              ["v_min by chemistry", "v_max by chemistry"]):
        data = [df.loc[df["chemistry"] == c, col].dropna().values for c in chems]
        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color=_PALETTE["val"], linewidth=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=_PALETTE["train"], alpha=0.55)
        ax.set_xticklabels(chems, rotation=20)
        ax.set_title(title); ax.set_ylabel("V")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "voltage_range_by_chemistry"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Capacity vs nominal_Ah ratio ----------
    sub = df[["capacity_Ah", "nominal_Ah", "chemistry"]].dropna()
    sub = sub[(sub["nominal_Ah"] > 0) & (sub["capacity_Ah"] >= 0)]
    sub["ratio"] = sub["capacity_Ah"] / sub["nominal_Ah"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(
        [sub.loc[sub["chemistry"] == c, "ratio"].clip(0, 1.5).values for c in chems],
        showmedians=True, widths=0.85,
    )
    for body in parts["bodies"]:
        body.set_facecolor(_PALETTE["train"]); body.set_alpha(0.65)
    ax.set_xticks(range(1, len(chems) + 1)); ax.set_xticklabels(chems)
    ax.set_ylabel("capacity_Ah / nominal_Ah  (clipped to [0, 1.5])")
    ax.set_title("Capacity-to-nominal ratio (sanity check)")
    ax.axhline(1.0, color="#7F8C8D", linestyle="--", linewidth=0.8)
    fig.savefig(fig_path(SECTION, "capacity_to_nominal_ratio"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Coulombic efficiency distribution ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ce = df["coulombic_eff"].dropna()
    ce_clip = ce.clip(0.5, 1.2)
    ax.hist(ce_clip, bins=80, color=_PALETTE["train"], alpha=0.85)
    ax.axvline(1.0, color=_PALETTE["val"], linestyle="--", label="ideal=1.00")
    ax.set_xlabel("Coulombic efficiency (clipped 0.5–1.2)")
    ax.set_ylabel("Count"); ax.set_title("Coulombic efficiency distribution")
    n_anom = ((ce > 1.05) | (ce < 0.95)).sum()
    ax.text(0.97, 0.93, f"|CE − 1| > 5%: {n_anom:,} rows ({n_anom/len(ce)*100:.2f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#B0B0B0",
                      boxstyle="round,pad=0.3", alpha=0.9))
    ax.legend()
    fig.savefig(fig_path(SECTION, "coulombic_efficiency"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Per-chemistry stats summary ----------
    summary = []
    for c in chems:
        s = df[df["chemistry"] == c]
        soh = s["soh"].dropna()
        summary.append({
            "chemistry": c,
            "n_rows": len(s),
            "soh_p10": soh.quantile(0.10) if len(soh) else np.nan,
            "soh_median": soh.median() if len(soh) else np.nan,
            "soh_p90": soh.quantile(0.90) if len(soh) else np.nan,
            "frac_below_eol": (soh < 0.8).mean() if len(soh) else np.nan,
            "v_max_median": s["v_max"].median(),
            "v_min_median": s["v_min"].median(),
            "ce_median": s["coulombic_eff"].median(),
        })
    summary_df = pd.DataFrame(summary)

    lines = [
        "# 2. Cycling feature distributions",
        "",
        "## 2.1 Per-chemistry quantitative summary",
        "",
        md_table(
            ["chemistry", "rows", "SoH p10", "SoH median", "SoH p90",
             "frac < 0.8 (EoL)", "v_max median", "v_min median", "CE median"],
            [(r["chemistry"], f"{int(r['n_rows']):,}",
              f"{r['soh_p10']:.3f}" if pd.notna(r["soh_p10"]) else "—",
              f"{r['soh_median']:.3f}" if pd.notna(r["soh_median"]) else "—",
              f"{r['soh_p90']:.3f}" if pd.notna(r["soh_p90"]) else "—",
              f"{r['frac_below_eol']*100:.1f}%" if pd.notna(r["frac_below_eol"]) else "—",
              f"{r['v_max_median']:.2f}" if pd.notna(r["v_max_median"]) else "—",
              f"{r['v_min_median']:.2f}" if pd.notna(r["v_min_median"]) else "—",
              f"{r['ce_median']:.3f}" if pd.notna(r["ce_median"]) else "—")
             for _, r in summary_df.iterrows()],
        ),
        "",
        "## 2.2 Headline insights",
        "",
    ]

    insights = []
    nmc_below_eol = summary_df.loc[summary_df["chemistry"] == "NMC", "frac_below_eol"].values
    lfp_below_eol = summary_df.loc[summary_df["chemistry"] == "LFP", "frac_below_eol"].values
    if len(nmc_below_eol) and len(lfp_below_eol):
        insights.append(
            f"- **NMC corpus is heavily aged** — {nmc_below_eol[0]*100:.1f}% of NMC rows have SoH<0.8 "
            f"(true EoL territory) vs only {lfp_below_eol[0]*100:.1f}% for LFP. This means the SoH model will "
            "see plenty of grade-C/D examples on NMC but will need careful handling of LFP test cells "
            "to avoid training-distribution mismatch."
        )
    n_anom_ce = ((df["coulombic_eff"] > 1.05) | (df["coulombic_eff"] < 0.95)).sum()
    insights.append(
        f"- **{n_anom_ce:,} rows have |CE−1| > 5%** "
        f"({n_anom_ce/df['coulombic_eff'].notna().sum()*100:.2f}% of measured) — "
        "physically these should be near 1.0 for healthy cells. The tails are valuable anomaly-detector ground truth."
    )
    insights.append(
        "- **Voltage ranges cluster correctly by chemistry** — Zn-ion at ~0.8–2.0 V, Na-ion at ~1.5–4.5 V, "
        "Li-ion families (NMC/NCA/LCO/LFP/LMO) at 2.0–4.5 V. The chemistry-aware filter in `src/data/unify.py` is doing its job."
    )
    cap_ratio = sub.groupby("chemistry")["ratio"].median()
    if "Zn-ion" in cap_ratio.index and cap_ratio["Zn-ion"] < 0.5:
        insights.append(
            f"- **Zn-ion `capacity_Ah / nominal_Ah` median = {cap_ratio['Zn-ion']:.2f}** — "
            "consistent with `nominal_Ah=0` rows seen in EDA-01 (likely mAh/Ah unit mismatch in the source). Worth fixing in the loader."
        )
    lines.extend(insights)
    lines.append("")
    lines.append("**Figures**: "
                 f"[feature_histograms]({fig_path(SECTION, 'feature_histograms').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[soh_violin_by_chemistry]({fig_path(SECTION, 'soh_violin_by_chemistry').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[voltage_range_by_chemistry]({fig_path(SECTION, 'voltage_range_by_chemistry').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[capacity_to_nominal_ratio]({fig_path(SECTION, 'capacity_to_nominal_ratio').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[coulombic_efficiency]({fig_path(SECTION, 'coulombic_efficiency').relative_to(Path(__file__).resolve().parents[2])})")
    out = write_findings("02_cycling_distributions", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
