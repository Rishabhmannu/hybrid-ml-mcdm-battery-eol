"""
Stage 10 — MCDM 5-scenario sensitivity sweep + RQ2 BWMR-vs-EU comparison.

Runs the canonical-criteria Fuzzy BWM-TOPSIS over 5 weight scenarios for each
of the 4 health grades (A/B/C/D), computes pairwise Spearman ρ stability
across all scenarios, and answers the RQ2 hypothesis (H2) by reporting
whether BWMR-Heavy and EU-Heavy weightings produce different route rankings.

Outputs (results/):
- tables/mcdm_sensitivity/rankings_long.csv    per-scenario × per-grade × per-alt
- tables/mcdm_sensitivity/recommendations.csv  rank-1 route per (scenario, grade)
- tables/mcdm_sensitivity/rho_matrix_mean.csv  pairwise stability matrix
- tables/mcdm_sensitivity/bwmr_vs_eu.json      RQ2 answer
- tables/mcdm_sensitivity/findings.md          executive summary
- figures/mcdm_sensitivity/rho_heatmap.png     stability heatmap
- figures/mcdm_sensitivity/recommendations_heatmap.png   route × grade × scenario
- figures/mcdm_sensitivity/bwmr_vs_eu_per_grade.png      RQ2 visualization
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mcdm.sensitivity import run_sensitivity
from src.mcdm.topsis import CANONICAL_ALTERNATIVES, CANONICAL_CRITERIA
from src.utils.config import RESULTS_DIR
from src.utils.plots import _PALETTE, apply_theme

OUT_TBL = RESULTS_DIR / "tables" / "mcdm_sensitivity"
OUT_FIG = RESULTS_DIR / "figures" / "mcdm_sensitivity"


def _save_rho_heatmap(rho: pd.DataFrame, out_path: Path, title: str):
    apply_theme()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(rho.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho.columns)))
    ax.set_xticklabels(rho.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(rho.index)))
    ax.set_yticklabels(rho.index)
    ax.set_title(title)
    for i in range(len(rho.index)):
        for j in range(len(rho.columns)):
            v = rho.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if abs(v) > 0.4 else "#555555", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_recommendations_heatmap(recos: pd.DataFrame, out_path: Path):
    apply_theme()
    alt_to_idx = {a: i for i, a in enumerate(CANONICAL_ALTERNATIVES)}
    matrix = recos.map(lambda a: alt_to_idx[a]).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10", len(CANONICAL_ALTERNATIVES))
    ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=len(CANONICAL_ALTERNATIVES) - 0.5,
              aspect="auto")
    ax.set_xticks(range(len(recos.columns)))
    ax.set_xticklabels([f"Grade {g}" for g in recos.columns])
    ax.set_yticks(range(len(recos.index)))
    ax.set_yticklabels(recos.index)
    ax.set_title("Rank-1 recommended route per (scenario × grade)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(recos.iat[i, j]), ha="center", va="center",
                    fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1.5,
                              edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_bwmr_vs_eu_per_grade(result: dict, out_path: Path):
    apply_theme()
    g_data = result["bwmr_eu_per_grade"]
    grades = list(g_data.keys())
    rhos = [g_data[g]["spearman_rho"] for g in grades]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [_PALETTE["test"] if r >= 0.8 else _PALETTE["accent"] if r >= 0.0
              else _PALETTE["val"] for r in rhos]
    bars = ax.bar([f"Grade {g}" for g in grades], rhos, color=colors, alpha=0.9)
    for bar, g, rho in zip(bars, grades, rhos):
        agree = g_data[g]["agree"]
        bwmr = g_data[g]["bwmr_recommended"]
        eu = g_data[g]["eu_recommended"]
        label = f"ρ={rho:+.2f}\n{'agree' if agree else 'differ'}\nBWMR: {bwmr}\nEU:   {eu}"
        ax.text(bar.get_x() + bar.get_width() / 2, max(rho, 0) + 0.05,
                label, ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, color="#7F8C8D", linewidth=0.7)
    ax.axhline(result["h2_threshold"], color="#7F8C8D", linestyle="--", linewidth=0.7,
               label=f"H2 falsification threshold ρ = {result['h2_threshold']}")
    ax.axhline(result["stability_target"], color="#7F8C8D", linestyle=":",
               linewidth=0.7, label=f"stability target ρ = {result['stability_target']}")
    ax.set_ylim(-1.05, 1.6)
    ax.set_ylabel("Spearman ρ (BWMR-Heavy vs EU-Heavy)")
    ax.set_title("RQ2 — BWMR vs EU rank stability per grade")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_findings(result: dict, out_path: Path):
    lines = []
    lines.append("# MCDM Sensitivity Sweep — Findings")
    lines.append("")
    lines.append(f"_Five scenarios over the canonical 6 criteria "
                 f"({', '.join(CANONICAL_CRITERIA)}), four grades (A/B/C/D)._")
    lines.append("")
    lines.append("## Scenario weight vectors")
    lines.append("")
    lines.append("| Scenario | " + " | ".join(CANONICAL_CRITERIA) + " |")
    lines.append("|" + "|".join(["---"] * (len(CANONICAL_CRITERIA) + 1)) + "|")
    for sc, w in result["scenarios"].items():
        lines.append(f"| {sc} | " + " | ".join(f"{x:.3f}" for x in w) + " |")
    lines.append("")

    lines.append("## Rank-1 recommended route per (scenario × grade)")
    lines.append("")
    recos = result["recommendations"]
    lines.append("| Scenario | Grade A | Grade B | Grade C | Grade D |")
    lines.append("|---|---|---|---|---|")
    for sc in recos.index:
        lines.append(f"| {sc} | " + " | ".join(recos.loc[sc].values) + " |")
    lines.append("")

    lines.append("## Pairwise stability (mean Spearman ρ across grades)")
    lines.append("")
    rho = result["rho_matrix_mean"]
    lines.append("| | " + " | ".join(rho.columns) + " |")
    lines.append("|" + "|".join(["---"] * (len(rho.columns) + 1)) + "|")
    for s in rho.index:
        lines.append(f"| **{s}** | " + " | ".join(f"{x:+.2f}" for x in rho.loc[s].values) + " |")
    lines.append("")
    lines.append(f"- Overall lowest off-diagonal ρ: **{result['overall_min_rho']:+.3f}**")
    lines.append(f"- Stability target (TARGETS.topsis_stability): {result['stability_target']:.2f}")
    lines.append("")

    lines.append("## RQ2 — BWMR-Heavy vs EU-Heavy answer")
    lines.append("")
    lines.append(f"**Mean Spearman ρ across the 4 grades: {result['mean_bwmr_eu_rho']:+.3f}**")
    lines.append(f"H2 falsification threshold: ρ < {result['h2_threshold']}")
    lines.append("")
    lines.append("| Grade | ρ (BWMR vs EU) | BWMR-Heavy recommends | EU-Heavy recommends | Agreement |")
    lines.append("|---|---|---|---|---|")
    for g, d in result["bwmr_eu_per_grade"].items():
        agree = "yes" if d["agree"] else "**no**"
        lines.append(f"| {g} | {d['spearman_rho']:+.2f} | {d['bwmr_recommended']} | {d['eu_recommended']} | {agree} |")
    lines.append("")
    if result["h2_rejected"]:
        lines.append("**Verdict: H2 rejected** — mean ρ falls below the falsification threshold. "
                     "BWMR and EU regulatory regimes produce convergent recommendations under our literature-derived "
                     "decision matrices, suggesting that the *current* MCDM framework is not sensitive to the "
                     "BWMR-vs-EU regime choice. This is itself a policy-relevant finding: criterion-weighting alone "
                     "may not be sufficient to differentiate Indian vs European EoL strategies — the divergence likely "
                     "lives in the underlying decision-matrix scores (which encode local cost/recovery realities) "
                     "rather than in the criterion weights.")
    else:
        agree_count = sum(1 for d in result["bwmr_eu_per_grade"].values() if d["agree"])
        lines.append(f"**Verdict: H2 supported** — BWMR and EU rankings differ on {4 - agree_count}/4 grades; "
                     f"mean ρ = {result['mean_bwmr_eu_rho']:+.3f} stays above the {result['h2_threshold']} "
                     f"falsification threshold. The framework's recommendation does respond to regulatory regime, "
                     "as the working hypothesis predicted.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MCDM 5-scenario sensitivity sweep + RQ2 comparison")
    print("=" * 70)

    result = run_sensitivity()

    # --- save tables ---
    result["rankings"].to_csv(OUT_TBL / "rankings_long.csv", index=False)
    result["recommendations"].to_csv(OUT_TBL / "recommendations.csv")
    result["rho_matrix_mean"].to_csv(OUT_TBL / "rho_matrix_mean.csv")
    for g, df in result["rho_matrix_per_grade"].items():
        df.to_csv(OUT_TBL / f"rho_matrix_grade_{g}.csv")
    with open(OUT_TBL / "bwmr_vs_eu.json", "w") as f:
        json.dump({
            "mean_spearman_rho": result["mean_bwmr_eu_rho"],
            "h2_falsification_threshold": result["h2_threshold"],
            "h2_rejected": result["h2_rejected"],
            "per_grade": result["bwmr_eu_per_grade"],
        }, f, indent=2)

    _write_findings(result, OUT_TBL / "findings.md")

    # --- save figures ---
    _save_rho_heatmap(result["rho_matrix_mean"],
                      OUT_FIG / "rho_heatmap.png",
                      "MCDM scenario stability — mean Spearman ρ across grades")
    _save_recommendations_heatmap(result["recommendations"],
                                  OUT_FIG / "recommendations_heatmap.png")
    _save_bwmr_vs_eu_per_grade(result, OUT_FIG / "bwmr_vs_eu_per_grade.png")

    # --- print headlines ---
    recos = result["recommendations"]
    print("\nRecommended route per (scenario × grade):")
    print(recos.to_string())
    print("\nPairwise mean Spearman ρ:")
    print(result["rho_matrix_mean"].round(3).to_string())
    print(f"\nOverall lowest off-diagonal ρ: {result['overall_min_rho']:+.3f}  "
          f"(stability target {result['stability_target']:.2f})")
    print(f"\nRQ2 — BWMR vs EU mean ρ across grades: {result['mean_bwmr_eu_rho']:+.3f}")
    print(f"H2 falsification threshold: ρ < {result['h2_threshold']}")
    print(f"H2 verdict: {'REJECTED' if result['h2_rejected'] else 'supported'}")
    agree = sum(1 for d in result["bwmr_eu_per_grade"].values() if d["agree"])
    print(f"BWMR/EU agreement on rank-1: {agree}/4 grades")
    for g, d in result["bwmr_eu_per_grade"].items():
        print(f"  Grade {g}: ρ={d['spearman_rho']:+.2f}  "
              f"BWMR→{d['bwmr_recommended']}  EU→{d['eu_recommended']}  "
              f"{'agree' if d['agree'] else '*differ*'}")

    rel_tbl = OUT_TBL.relative_to(PROJECT_ROOT)
    rel_fig = OUT_FIG.relative_to(PROJECT_ROOT)
    print(f"\nTables → {rel_tbl}/")
    print(f"Figures → {rel_fig}/")
    print("Done.")


if __name__ == "__main__":
    main()
