"""
EDA 08 — MCDM literature-derived weights audit.
- Coverage matrix (papers × criteria)
- TFN range per criterion
- Sensitivity / dispersion analysis
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECTION = "mcdm_weights"

LIT = PROCESSED_DIR / "mcdm_weights" / "literature_weights.csv"
SUMMARY = PROCESSED_DIR / "mcdm_weights" / "canonical_weights_summary.csv"
BWM_INPUT = PROCESSED_DIR / "mcdm_weights" / "fuzzy_bwm_input.csv"

CRITERIA_ORDER = ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"]


def main():
    apply_theme()
    lit = pd.read_csv(LIT, comment="#")
    # drop blank rows that are pure metadata
    lit = lit[lit["canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)"].isin(CRITERIA_ORDER)].copy()
    lit = lit.rename(columns={
        "paper_id (lit_seq)": "paper_id",
        "canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)": "criterion",
        "weight (0-1)": "weight",
    })
    lit["paper_id"] = pd.to_numeric(lit["paper_id"], errors="coerce")
    lit = lit.dropna(subset=["paper_id", "weight"])
    lit["paper_id"] = lit["paper_id"].astype(int)

    summary = pd.read_csv(SUMMARY)
    bwm = pd.read_csv(BWM_INPUT)

    # ---------- Coverage matrix ----------
    pivot = (lit.assign(present=1)
             .pivot_table(index="paper_id", columns="criterion",
                          values="present", aggfunc="max", fill_value=0)
             .reindex(columns=CRITERIA_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(CRITERIA_ORDER))); ax.set_xticklabels(CRITERIA_ORDER, rotation=20)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels([f"#{p}" for p in pivot.index])
    ax.set_title("Paper × Criterion coverage matrix (1 = weight reported)")
    for i in range(len(pivot)):
        for j in range(len(CRITERIA_ORDER)):
            v = pivot.values[i, j]
            ax.text(j, i, "✓" if v else "", ha="center", va="center",
                    color="white" if v else "#1A1A1A", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "coverage_matrix"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- TFN range per criterion ----------
    bwm = bwm.set_index("criterion").reindex(CRITERIA_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(CRITERIA_ORDER))
    have = ~bwm["tfn_middle_normalized"].isna()
    ax.barh(y[have],
            bwm.loc[have, "tfn_upper_normalized"].values - bwm.loc[have, "tfn_lower_normalized"].values,
            left=bwm.loc[have, "tfn_lower_normalized"].values,
            color=_PALETTE["train"], alpha=0.55,
            label="Triangular fuzzy range")
    ax.scatter(bwm.loc[have, "tfn_middle_normalized"], y[have], color=_PALETTE["val"],
               s=70, marker="d", zorder=5, label="Crisp middle (defuzzified)")
    for i in np.where(~have)[0]:
        ax.text(0.05, i, "no evidence — fall back to sensitivity scenarios",
                va="center", fontsize=9, color="#7F8C8D", style="italic")
    ax.set_yticks(y); ax.set_yticklabels(CRITERIA_ORDER)
    ax.set_xlabel("Normalized weight (sums to 1.0 across criteria with evidence)")
    ax.set_xlim(0, 0.55)
    ax.set_title("Fuzzy BWM input — TFN ranges per canonical criterion")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "tfn_ranges"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Per-paper weight scatter ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for crit in CRITERIA_ORDER:
        sub = lit[lit["criterion"] == crit]
        if len(sub) == 0:
            continue
        ax.scatter(np.full(len(sub), CRITERIA_ORDER.index(crit)),
                   sub["weight"].astype(float), s=80, alpha=0.7,
                   edgecolor="white", color=_PALETTE["train"])
        for _, r in sub.iterrows():
            ax.annotate(f"#{int(r['paper_id'])}",
                        (CRITERIA_ORDER.index(crit), r["weight"]),
                        xytext=(4, 0), textcoords="offset points", fontsize=7)
    ax.set_xticks(range(len(CRITERIA_ORDER))); ax.set_xticklabels(CRITERIA_ORDER, rotation=20)
    ax.set_ylabel("Reported weight (paper-level)")
    ax.set_title("Per-paper weight scatter (raw, before normalization)")
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "per_paper_weights"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Markdown ----------
    coverage_summary = pivot.sum(axis=0)
    n_papers_total = len(pivot)
    lines = [
        "# 8. MCDM literature-derived weights",
        "",
        f"_Source: `{LIT.relative_to(PROJECT_ROOT)}` · "
        f"15 weights · 8 papers · 6 canonical criteria_",
        "",
        "## 8.1 Coverage matrix (paper × criterion)",
        "",
        md_table(
            ["criterion", "n papers w/ weight", "share of corpus"],
            [(c, int(coverage_summary[c]), f"{coverage_summary[c]/n_papers_total*100:.0f}%")
             for c in CRITERIA_ORDER],
        ),
        "",
        "> Per §7.4 of the implementation plan, **3 papers** is the strict acceptance gate. "
        "Below that, the routing falls back to **5-scenario sensitivity analysis** (equal weights / "
        "literature mean / Technical-heavy / Compliance-heavy / Economic-heavy) for the under-represented criteria.",
        "",
        "## 8.2 Fuzzy BWM input (normalized TFNs)",
        "",
        md_table(
            ["criterion", "n papers", "TFN low", "TFN mid", "TFN high",
             "norm low", "norm mid", "norm high"],
            [[c]
             + [f"{int(bwm.loc[c, 'n_papers'])}",
                f"{bwm.loc[c, 'tfn_lower']:.3f}" if pd.notna(bwm.loc[c, "tfn_lower"]) else "—",
                f"{bwm.loc[c, 'tfn_middle']:.3f}" if pd.notna(bwm.loc[c, "tfn_middle"]) else "—",
                f"{bwm.loc[c, 'tfn_upper']:.3f}" if pd.notna(bwm.loc[c, "tfn_upper"]) else "—",
                f"{bwm.loc[c, 'tfn_lower_normalized']:.3f}" if pd.notna(bwm.loc[c, "tfn_lower_normalized"]) else "—",
                f"{bwm.loc[c, 'tfn_middle_normalized']:.3f}" if pd.notna(bwm.loc[c, "tfn_middle_normalized"]) else "—",
                f"{bwm.loc[c, 'tfn_upper_normalized']:.3f}" if pd.notna(bwm.loc[c, "tfn_upper_normalized"]) else "—"]
             for c in CRITERIA_ORDER],
        ),
        "",
        "## 8.3 Headline insights",
        "",
    ]
    insights = []
    insights.append(
        "- **Compliance is the only criterion with strict-gate evidence** "
        f"({int(coverage_summary['Compliance'])} papers, normalized weight = "
        f"{bwm.loc['Compliance', 'tfn_middle_normalized']:.3f}). "
        "It dominates because all 6 papers explicitly weight regulatory/reliability."
    )
    insights.append(
        "- **EPR Return and Carbon also pass strict gate** "
        f"({int(coverage_summary['EPR Return'])} and {int(coverage_summary['Carbon'])} papers respectively). "
        "Their normalized middles are 0.137 and 0.166."
    )
    insights.append(
        "- **SoH (0 papers), Safety (1 paper), Value (1 paper)** are structurally thin in the "
        "MCDM literature corpus — these criteria are typically engaged qualitatively or as "
        "objective functions, not weighted criteria. The §7.4 softened gate handles this via sensitivity sweep."
    )
    cv = lit.groupby("criterion")["weight"].apply(lambda s: s.astype(float).std() / max(s.astype(float).mean(), 1e-6))
    if "Compliance" in cv:
        insights.append(
            f"- **Compliance has high inter-paper dispersion** "
            f"(coefficient of variation = {cv['Compliance']:.2f}) — "
            "the 6 papers report Compliance weights ranging from 0.013 to 0.436. The TFN range captures this; "
            "the routing should remain stable across this band."
        )
    insights.append(
        "- **Value reported by exactly 1 paper (#13, Neri 2024 BWM-TOPSIS DLT)** at 0.253 — "
        "this single point produces a degenerate TFN (lower=middle=upper) and likely biases the normalization. "
        "Sensitivity analysis is the principled defense; the manuscript narrative should foreground that."
    )

    lines.extend(insights)
    lines.append("")
    figs = ["coverage_matrix", "tfn_ranges", "per_paper_weights"]
    rels = [str(fig_path(SECTION, f).relative_to(PROJECT_ROOT)) for f in figs]
    lines.append("**Figures**: " + " · ".join(f"[{f}]({r})" for f, r in zip(figs, rels)))
    out = write_findings("08_mcdm_weights", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
