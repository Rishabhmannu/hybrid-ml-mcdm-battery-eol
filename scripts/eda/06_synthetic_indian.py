"""
EDA 06 — Synthetic Indian-context cells (PyBaMM OKane2022 NMC + Prada2013 LFP).
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
SECTION = "synthetic_indian"

THERMAL_ORDER = ["Bengaluru_mild", "Mumbai_monsoon", "Delhi_summer", "Rajasthan_extreme"]


def main():
    apply_theme()
    print(f"Loading {UNIFIED.relative_to(PROCESSED_DIR.parent.parent)} ...")
    df = pd.read_parquet(UNIFIED)
    syn = df[df["source"].str.startswith("SYN_IN", na=False)].copy()
    syn["thermal"] = syn["source"].str.replace("SYN_IN_NMC_", "", regex=False) \
                                  .str.replace("SYN_IN_LFP_", "", regex=False)
    syn["chem"] = np.where(syn["source"].str.contains("_NMC_"), "NMC",
                  np.where(syn["source"].str.contains("_LFP_"), "LFP", "other"))
    print(f"  synthetic rows: {len(syn):,}  unique cells: {syn['battery_id'].nunique()}")

    if len(syn) == 0:
        out = write_findings("06_synthetic_indian",
                             ["# 6. Synthetic Indian cells", "", "_No synthetic cells found in unified.parquet._"])
        print("  no synthetic data, skipping plots")
        return

    # ---------- Per-cell SoH curve sample ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, chem in zip(axes, ["NMC", "LFP"]):
        chem_sub = syn[syn["chem"] == chem]
        for thermal in THERMAL_ORDER:
            t_sub = chem_sub[chem_sub["thermal"] == thermal]
            cells = t_sub["battery_id"].unique()
            if len(cells) == 0:
                continue
            cell = cells[0]
            cd = t_sub[t_sub["battery_id"] == cell].sort_values("cycle")
            ax.plot(cd["cycle"], cd["soh"], label=thermal, linewidth=1.4)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH")
        ax.set_title(f"Synthetic {chem} — sample cell per thermal profile")
        ax.legend(fontsize=8); ax.set_ylim(0.96, 1.005) if chem == "LFP" else ax.set_ylim(0.85, 1.005)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "soh_per_thermal_profile"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Capacity fade rate per (chem, thermal) ----------
    rates = []
    for bid, g in syn.groupby("battery_id"):
        g = g.sort_values("cycle")
        if len(g) < 5:
            continue
        slope = np.polyfit(g["cycle"], g["soh"], 1)[0]
        rates.append({
            "battery_id": bid,
            "thermal": g["thermal"].iloc[0],
            "chem": g["chem"].iloc[0],
            "fade_pct_per_100cyc": slope * 100 * 100,
        })
    rdf = pd.DataFrame(rates)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, chem in zip(axes, ["NMC", "LFP"]):
        sub = rdf[rdf["chem"] == chem]
        order = [t for t in THERMAL_ORDER if t in sub["thermal"].unique()]
        data = [sub.loc[sub["thermal"] == t, "fade_pct_per_100cyc"].values for t in order]
        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color=_PALETTE["val"], linewidth=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=_PALETTE["train"], alpha=0.55)
        ax.set_xticklabels(order, rotation=15, fontsize=9)
        ax.set_ylabel("Fade rate (% SoH per 100 cycles)")
        ax.set_title(f"Synthetic {chem} — fade rate by thermal profile")
        ax.axhline(0, color="#7F8C8D", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "fade_per_thermal_profile"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Median fade per (chem, thermal) bar ----------
    pivot = (rdf.groupby(["chem", "thermal"])["fade_pct_per_100cyc"]
             .median().unstack().reindex(columns=THERMAL_ORDER))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pivot.T.plot(kind="bar", ax=ax, color=[_PALETTE["train"], _PALETTE["val"]], width=0.7)
    ax.set_ylabel("Median fade rate (% SoH per 100 cyc)")
    ax.set_title("Thermal-stress ordering check (synthetic only)")
    ax.set_xticklabels(THERMAL_ORDER, rotation=15)
    ax.legend(title="chemistry")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "thermal_stress_ordering"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Synthetic vs real reference cell SoH curves ----------
    ref_id = "BL_SNL_SNL_18650_NMC_25C_0-100_0.5-2C_b"
    if ref_id in df["battery_id"].values:
        ref = df[df["battery_id"] == ref_id].sort_values("cycle")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        # take a Delhi NMC synthetic cell as representative
        cands = syn.loc[(syn["chem"] == "NMC") & (syn["thermal"] == "Delhi_summer"), "battery_id"].unique()
        if len(cands) > 0:
            sc = syn[syn["battery_id"] == cands[0]].sort_values("cycle")
            ax.plot(sc["cycle"], sc["soh"], label=f"Synthetic NMC (Delhi) cell {cands[0][-12:]}",
                    color=_PALETTE["accent"], linewidth=1.4)
        ax.plot(ref["cycle"], ref["soh"], label=f"Real SNL NMC reference",
                color=_PALETTE["train"], linewidth=1.2, alpha=0.85)
        ax.axhline(0.8, color="#7F8C8D", linestyle="--", linewidth=0.7, label="EoL=0.80")
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH")
        ax.set_title("Synthetic vs real reference cell — SoH trajectories")
        ax.legend(fontsize=8)
        fig.savefig(fig_path(SECTION, "synthetic_vs_real_reference"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---------- Markdown ----------
    counts = (syn.groupby(["chem", "thermal"])["battery_id"].nunique()
              .unstack(fill_value=0).reindex(columns=THERMAL_ORDER))

    lines = [
        "# 6. Synthetic Indian-context cells",
        "",
        f"_Source: PyBaMM regenerated 2026-04-28 — OKane2022 + canonical degradation for NMC, "
        f"Prada2013 + isothermal SPMe for LFP._",
        "",
        f"- Synthetic rows in unified.parquet: **{len(syn):,}**",
        f"- Synthetic cells: **{syn['battery_id'].nunique()}**",
        f"- Sources: **{syn['source'].nunique()}** "
        f"({len(syn['chem'].unique())} chemistries × {len(syn['thermal'].unique())} thermal profiles)",
        "",
        "## 6.1 Cell counts per (chemistry × thermal profile)",
        "",
        md_table(
            ["chem"] + THERMAL_ORDER + ["total"],
            [[chem] + [int(counts.loc[chem, t]) if t in counts.columns and chem in counts.index else 0
                       for t in THERMAL_ORDER]
                    + [int(counts.loc[chem].sum()) if chem in counts.index else 0]
             for chem in counts.index],
        ),
        "",
        "## 6.2 Median fade rate (% SoH per 100 cycles)",
        "",
        md_table(
            ["chem"] + THERMAL_ORDER,
            [[chem] + [f"{pivot.loc[chem, t]:.3f}" if chem in pivot.index and t in pivot.columns and pd.notna(pivot.loc[chem, t]) else "—"
                       for t in THERMAL_ORDER]
             for chem in pivot.index],
        ),
        "",
        "## 6.3 Headline insights",
        "",
    ]

    insights = []
    nmc_row = pivot.loc["NMC"] if "NMC" in pivot.index else None
    if nmc_row is not None and nmc_row.notna().any():
        order_check = list(nmc_row.dropna().sort_values().index)
        expected = [t for t in THERMAL_ORDER if t in order_check]
        ordering_ok = order_check == expected
        if ordering_ok:
            insights.append(
                "- ✅ **NMC fade-rate ordering matches thermal stress** "
                "(Bengaluru < Mumbai < Delhi < Rajasthan) — confirms PyBaMM thermal coupling is wired correctly."
            )
        else:
            insights.append(
                f"- ⚠ **NMC fade-rate ordering deviates**: observed {order_check}. "
                f"Expected {expected}. Check the Bengaluru thermal profile parameters."
            )
    if "LFP" in pivot.index:
        lfp_row = pivot.loc["LFP"].dropna()
        if (lfp_row.abs() < 0.01).all():
            insights.append(
                "- **LFP fade rates are ~0** — expected because LFP synthetic cells run isothermal SPMe "
                "without degradation enabled (Prada2013 parameter set lacks SEI + current-collector params). "
                "Documented field-level limit, not a bug."
            )
    insights.append(
        "- **Synthetic cells are pinned to the train split** (anti-leakage rule §5.2 #3 in the implementation plan) — "
        "they augment training but do not contaminate val/test. Stage 9 SoH model evaluation reflects real-cell performance only."
    )
    lines.extend(insights)
    lines.append("")
    figs = ["soh_per_thermal_profile", "fade_per_thermal_profile",
            "thermal_stress_ordering", "synthetic_vs_real_reference"]
    rels = [str(fig_path(SECTION, f).relative_to(Path(__file__).resolve().parents[2])) for f in figs]
    lines.append("**Figures**: " + " · ".join(f"[{f}]({r})" for f, r in zip(figs, rels)))
    out = write_findings("06_synthetic_indian", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
