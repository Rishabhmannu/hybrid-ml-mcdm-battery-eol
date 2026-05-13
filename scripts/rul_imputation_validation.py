"""
Held-out validation of RUL imputation methods.

Strategy: take UNCENSORED cells (where we know the true EoL cycle), artificially
truncate each at multiple SoH thresholds to simulate censoring, run every
imputation method on the truncated trajectory, and measure imputed_eol_cycle
vs true_eol_cycle. This gives us a controlled experiment for which method
generalises best on which chemistries / sources / truncation depths.

Truncation thresholds simulate the censoring distribution observed in real
data (see `scripts/rul_diagnostic.py`):
- 0.85   — well above EoL (deeply censored, hardest to impute)
- 0.83   — moderately censored
- 0.81+ε — near-EoL (matches median min-SoH=0.817 of actual censored cells)

Outputs
-------
results/tables/rul_imputation_validation/
    summary.csv                  : one row per (method × truncation_soh × chemistry) cell
    aggregates_by_method.csv     : MAE / MAPE / convergence per method
    aggregates_by_method_chem.csv: same, stratified by chemistry
    aggregates_by_method_trunc.csv: same, stratified by truncation level
    findings.md                  : per-method ranking + recommended winner per chemistry

results/figures/rul_imputation_validation/
    mape_per_method.png
    error_distribution.png
    per_chemistry_heatmap.png

Usage
-----
    python scripts/rul_imputation_validation.py             # full corpus
    python scripts/rul_imputation_validation.py --smoke     # 50 cells, 3 methods
    python scripts/rul_imputation_validation.py --methods linear,exp2,kww,nn,gp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rul_imputation import (
    ALL_IMPUTERS, BaseImputer, CellTrajectory, EOL_THRESHOLD,
    cells_from_parquet, make_imputer,
)
from src.utils.config import PROCESSED_DIR, RESULTS_DIR

UNIFIED_PARQUET = PROCESSED_DIR / "cycling" / "unified.parquet"
OUT_TBL = RESULTS_DIR / "tables" / "rul_imputation_validation"
OUT_FIG = RESULTS_DIR / "figures" / "rul_imputation_validation"

# Truncation thresholds chosen to span the actual-censored cells' min-SoH range
# (median 0.817, ranges roughly 0.79-0.90).
DEFAULT_TRUNCATIONS = (0.85, 0.83, 0.815)

# Population-aware methods need fit() on a training subset.
POPULATION_AWARE = {"nn", "ml"}


# =============================================================================
# Validation harness
# =============================================================================

def filter_uncensored(cells: list[CellTrajectory],
                      min_n_for_validation: int = 30) -> list[CellTrajectory]:
    """Keep only uncensored cells with enough observations to support
    truncation + imputation."""
    out = []
    for c in cells:
        if c.is_censored:
            continue
        if c.n_observed < min_n_for_validation:
            continue
        if c.true_eol_cycle is None:
            continue
        # Need observations BEFORE the truncation threshold, ie cell must have
        # cycles where SoH > 0.85 (otherwise we can't truncate at 0.85).
        if c.soh.max() < 0.86:
            continue
        out.append(c)
    return out


def split_population_test(cells: list[CellTrajectory],
                          test_frac: float = 0.3,
                          seed: int = 42) -> tuple[list[CellTrajectory], list[CellTrajectory]]:
    """Split uncensored cells into a population (for fitting NN/ML imputers)
    and a held-out test set (for measuring imputation accuracy).
    Stratified by chemistry where possible."""
    chems = [c.chemistry for c in cells]
    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(cells)),
            test_size=test_frac,
            stratify=chems,
            random_state=seed,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            np.arange(len(cells)),
            test_size=test_frac,
            random_state=seed,
        )
    return [cells[i] for i in train_idx], [cells[i] for i in test_idx]


def evaluate_one(imputer: BaseImputer, cell: CellTrajectory,
                 truncation_soh: float) -> dict:
    """Truncate `cell` at `truncation_soh`, impute, and return a row of metrics
    to append to the summary table."""
    truncated = cell.truncate_at_soh(truncation_soh)
    if truncated.n_observed < 5:
        return None
    if truncated.soh.min() > truncation_soh + 0.01:
        return None  # truncation didn't actually trigger; skip
    if cell.true_eol_cycle is None:
        return None

    t0 = time.time()
    try:
        result = imputer.impute(truncated)
    except Exception as e:
        return {
            "method": imputer.name,
            "battery_id": cell.battery_id,
            "chemistry": cell.chemistry,
            "source": cell.source,
            "truncation_soh": truncation_soh,
            "true_eol_cycle": float(cell.true_eol_cycle),
            "imputed_eol_cycle": float("nan"),
            "converged": False,
            "abs_err_cycles": float("nan"),
            "rel_err_pct": float("nan"),
            "n_observed_after_trunc": int(truncated.n_observed),
            "min_soh_after_trunc": float(truncated.min_soh),
            "exception": str(e),
            "compute_s": time.time() - t0,
        }
    elapsed = time.time() - t0

    if (not result.converged) or np.isnan(result.imputed_eol_cycle):
        abs_err = float("nan"); rel_err = float("nan")
    else:
        abs_err = abs(result.imputed_eol_cycle - cell.true_eol_cycle)
        rel_err = abs_err / cell.true_eol_cycle * 100.0

    return {
        "method": imputer.name,
        "battery_id": cell.battery_id,
        "chemistry": cell.chemistry,
        "source": cell.source,
        "truncation_soh": truncation_soh,
        "true_eol_cycle": float(cell.true_eol_cycle),
        "imputed_eol_cycle": float(result.imputed_eol_cycle),
        "converged": bool(result.converged) and not np.isnan(result.imputed_eol_cycle),
        "abs_err_cycles": abs_err,
        "rel_err_pct": rel_err,
        "n_observed_after_trunc": int(truncated.n_observed),
        "min_soh_after_trunc": float(truncated.min_soh),
        "compute_s": elapsed,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Held-out validation of RUL imputation methods")
    p.add_argument("--smoke", action="store_true", help="50 test cells, 3 methods")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated subset of methods (default: all). "
                        f"Available: {sorted(ALL_IMPUTERS)}")
    p.add_argument("--truncations", type=str, default=None,
                   help="Comma-separated truncation SoH levels "
                        f"(default: {DEFAULT_TRUNCATIONS})")
    p.add_argument("--max-test-cells", type=int, default=None,
                   help="Cap test cells (default: no cap)")
    args = p.parse_args()

    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    methods = (args.methods.split(",") if args.methods
               else (["linear", "kww", "powerlaw"] if args.smoke
                     else list(ALL_IMPUTERS.keys())))
    methods = [m.strip() for m in methods]
    truncations = ([float(t) for t in args.truncations.split(",")]
                   if args.truncations else list(DEFAULT_TRUNCATIONS))

    print("=" * 75)
    print(f"RUL imputation held-out validation  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"  methods: {methods}")
    print(f"  truncations: {truncations}")
    print("=" * 75)

    # ---- Load corpus ----
    print(f"\n[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(UNIFIED_PARQUET)
    cells = cells_from_parquet(df, min_observed=20)
    print(f"  {len(cells):,} cells with ≥20 observed cycles")

    uncensored = filter_uncensored(cells, min_n_for_validation=30)
    print(f"  {len(uncensored):,} uncensored cells with sufficient cycles "
          f"and SoH range ≥0.86 (validation pool)")

    if args.smoke:
        rng = np.random.default_rng(42)
        smoke_idx = rng.choice(len(uncensored), size=min(50, len(uncensored)), replace=False)
        uncensored = [uncensored[i] for i in smoke_idx]
        print(f"  SMOKE: subsampled to {len(uncensored)} cells")

    population, test_cells = split_population_test(uncensored, test_frac=0.3, seed=42)
    if args.max_test_cells and len(test_cells) > args.max_test_cells:
        rng = np.random.default_rng(123)
        keep = rng.choice(len(test_cells), size=args.max_test_cells, replace=False)
        test_cells = [test_cells[i] for i in keep]
    print(f"\n[Split] {len(population):,} population cells / {len(test_cells):,} held-out test cells")
    chem_counts = pd.Series([c.chemistry for c in test_cells]).value_counts()
    print(f"  test chemistries: {chem_counts.to_dict()}")

    # ---- Fit population-aware imputers ONCE ----
    print(f"\n[Fit] Fitting imputers on {len(population)} population cells ...")
    fitted: dict[str, BaseImputer] = {}
    for m in methods:
        try:
            imp = make_imputer(m)
            t0 = time.time()
            imp.fit(population)
            print(f"  {m:14s}  fit time {time.time()-t0:.2f}s")
            fitted[m] = imp
        except Exception as e:
            print(f"  {m:14s}  FIT FAILED: {e}")

    # ---- Run validation ----
    rows = []
    print(f"\n[Validate] {len(test_cells)} cells × {len(truncations)} truncations × {len(fitted)} methods")
    print(f"  ≈ {len(test_cells)*len(truncations)*len(fitted):,} total imputations")
    t_start = time.time()
    for m, imp in fitted.items():
        m_t0 = time.time()
        n_done = 0; n_converged = 0
        for cell in test_cells:
            for trunc in truncations:
                row = evaluate_one(imp, cell, trunc)
                if row is None:
                    continue
                rows.append(row)
                n_done += 1
                if row["converged"]:
                    n_converged += 1
        print(f"  {m:14s}  {n_done:>5d} runs  {n_converged:>5d} converged "
              f"({n_converged/max(n_done,1)*100:5.1f}%)  in {time.time()-m_t0:.1f}s")
    print(f"  Total wall-clock: {time.time()-t_start:.1f}s")

    if not rows:
        print("\n[ERROR] No results — bailing.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_TBL / "summary.csv", index=False)
    print(f"\n[Save] {(OUT_TBL / 'summary.csv').relative_to(PROJECT_ROOT)}  ({len(summary):,} rows)")

    # ---- Aggregates ----
    def _agg(df_sub: pd.DataFrame) -> pd.Series:
        valid = df_sub[df_sub["converged"]]
        return pd.Series({
            "n_runs":       int(len(df_sub)),
            "n_converged":  int(len(valid)),
            "convergence_rate":  float(len(valid) / max(len(df_sub), 1)),
            "mae_cycles":   float(valid["abs_err_cycles"].mean()) if len(valid) else float("nan"),
            "median_abs_err_cycles": float(valid["abs_err_cycles"].median()) if len(valid) else float("nan"),
            "mape_pct":     float(valid["rel_err_pct"].mean()) if len(valid) else float("nan"),
            "median_rel_err_pct": float(valid["rel_err_pct"].median()) if len(valid) else float("nan"),
            "p90_rel_err_pct": float(valid["rel_err_pct"].quantile(0.90)) if len(valid) else float("nan"),
            "compute_s_per_cell": float(df_sub["compute_s"].mean()),
        })

    by_method = summary.groupby("method", as_index=True).apply(
        _agg, include_groups=False
    ).sort_values("median_rel_err_pct")
    by_method.to_csv(OUT_TBL / "aggregates_by_method.csv")
    print(f"\n[Aggregate] By method:")
    cols = ["n_runs", "convergence_rate", "median_rel_err_pct", "mape_pct",
            "p90_rel_err_pct", "compute_s_per_cell"]
    print(by_method[cols].to_string(float_format="%.3f"))

    by_method_chem = summary.groupby(["method", "chemistry"], as_index=True).apply(
        _agg, include_groups=False
    )
    by_method_chem.to_csv(OUT_TBL / "aggregates_by_method_chem.csv")

    by_method_trunc = summary.groupby(["method", "truncation_soh"], as_index=True).apply(
        _agg, include_groups=False
    )
    by_method_trunc.to_csv(OUT_TBL / "aggregates_by_method_trunc.csv")

    # ---- Per-chemistry winner ----
    chem_winners = {}
    for chem in summary["chemistry"].unique():
        sub = by_method_chem.xs((slice(None), chem),
                                level=("method", "chemistry"),
                                drop_level=False)
        eligible = sub[sub["convergence_rate"] >= 0.5].sort_values("median_rel_err_pct")
        if len(eligible):
            winner = eligible.index[0][0]  # method name
            chem_winners[chem] = {
                "method": winner,
                "median_rel_err_pct": float(eligible.iloc[0]["median_rel_err_pct"]),
                "convergence_rate": float(eligible.iloc[0]["convergence_rate"]),
            }
        else:
            chem_winners[chem] = None
    print(f"\n[Winner per chemistry]")
    for chem, w in chem_winners.items():
        if w:
            print(f"  {chem:10s}  → {w['method']:14s}  "
                  f"median rel err {w['median_rel_err_pct']:5.2f}%  "
                  f"conv {w['convergence_rate']*100:5.1f}%")
        else:
            print(f"  {chem:10s}  → no method passed convergence threshold")

    with open(OUT_TBL / "chemistry_winners.json", "w") as f:
        json.dump(chem_winners, f, indent=2)

    # ---- Figures ----
    try:
        _plot_per_method_box(summary)
        _plot_per_method_chem_heatmap(by_method_chem.reset_index())
        _plot_per_truncation_lines(by_method_trunc.reset_index())
    except Exception as e:
        print(f"[WARN] figure generation failed: {e}")

    _write_findings(by_method, chem_winners, by_method_trunc.reset_index(),
                    n_test=len(test_cells), n_pop=len(population),
                    methods=list(fitted.keys()), truncations=truncations)
    print(f"\nDone.")


# =============================================================================
# Plots
# =============================================================================

def _plot_per_method_box(summary: pd.DataFrame):
    valid = summary[summary["converged"]]
    if valid.empty:
        return
    methods_in_order = (valid.groupby("method")["rel_err_pct"]
                             .median().sort_values().index.tolist())
    data = [valid.loc[valid["method"] == m, "rel_err_pct"].values for m in methods_in_order]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, labels=methods_in_order, showfliers=False)
    ax.set_ylabel("Imputation rel. error (% of true EoL)")
    ax.set_xlabel("Imputation method (sorted by median error, ascending)")
    ax.set_title("Held-out RUL imputation accuracy — per method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "mape_per_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_per_method_chem_heatmap(by_method_chem_df: pd.DataFrame):
    pivot = by_method_chem_df.pivot(index="method", columns="chemistry",
                                    values="median_rel_err_pct")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}" if pd.notna(v) else "—",
                    ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Median rel. error (%) — method × chemistry (lower = better)")
    fig.colorbar(im, ax=ax, label="Median rel. err (%)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "per_chemistry_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_per_truncation_lines(by_method_trunc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in by_method_trunc_df["method"].unique():
        sub = by_method_trunc_df[by_method_trunc_df["method"] == m].sort_values("truncation_soh")
        ax.plot(sub["truncation_soh"], sub["median_rel_err_pct"],
                marker="o", label=m)
    ax.set_xlabel("Truncation SoH (artificial censoring threshold)")
    ax.set_ylabel("Median imputation rel. error (%)")
    ax.set_title("Imputation accuracy vs how-deeply-censored the cell is")
    ax.invert_xaxis()
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "error_vs_truncation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Findings markdown
# =============================================================================

def _write_findings(by_method: pd.DataFrame, chem_winners: dict,
                    by_method_trunc_df: pd.DataFrame,
                    n_test: int, n_pop: int,
                    methods: list[str], truncations: list[float]):
    by_method = by_method.copy()
    overall_winner = by_method.index[0]
    overall_median = float(by_method.iloc[0]["median_rel_err_pct"])

    lines = []
    lines.append("# RUL imputation held-out validation — findings\n")
    lines.append(f"_Generated by `scripts/rul_imputation_validation.py`._\n")
    lines.append(f"\n## Setup\n")
    lines.append(f"- Population (used to fit population-aware imputers): **{n_pop} uncensored cells**")
    lines.append(f"- Held-out test set: **{n_test} uncensored cells** (artificially truncated to simulate censoring)")
    lines.append(f"- Truncation SoH levels: **{truncations}**")
    lines.append(f"- Methods evaluated: **{methods}**\n")
    lines.append(f"## Overall ranking (sorted by median rel. err)\n\n")
    lines.append("| Rank | Method | Median rel. err (%) | MAPE (%) | P90 rel. err (%) | Convergence | Compute (s/cell) |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, (method, row) in enumerate(by_method.iterrows(), start=1):
        lines.append(f"| {i} | **{method}** | "
                     f"{row['median_rel_err_pct']:.2f} | "
                     f"{row['mape_pct']:.2f} | "
                     f"{row['p90_rel_err_pct']:.2f} | "
                     f"{row['convergence_rate']*100:.1f}% | "
                     f"{row['compute_s_per_cell']:.3f} |")
    lines.append(f"\n**Overall winner: `{overall_winner}` (median rel. err {overall_median:.2f}%)**.\n")

    lines.append(f"\n## Winner per chemistry\n")
    lines.append("| Chemistry | Best method | Median rel. err (%) | Convergence |")
    lines.append("|---|---|---|---|")
    for chem, w in chem_winners.items():
        if w:
            lines.append(f"| {chem} | **{w['method']}** | {w['median_rel_err_pct']:.2f} | "
                         f"{w['convergence_rate']*100:.1f}% |")
        else:
            lines.append(f"| {chem} | _no method passed_ | — | — |")

    lines.append(f"\n## Behaviour vs truncation depth\n")
    lines.append("(How does imputation accuracy degrade as we censor more aggressively? "
                 "Lower truncation SoH = harder problem, more extrapolation needed.)\n")
    pivot = by_method_trunc_df.pivot(index="method", columns="truncation_soh",
                                     values="median_rel_err_pct")
    pivot = pivot.reindex(by_method.index)  # keep ranking
    lines.append("| Method | " + " | ".join(f"trunc=SoH={t:.2f}" for t in pivot.columns) + " |")
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for m, row in pivot.iterrows():
        lines.append("| " + m + " | " + " | ".join(
            f"{v:.2f}" if pd.notna(v) else "—" for v in row.values
        ) + " |")

    lines.append(f"\n## Next-step recommendation\n")
    if overall_median < 5:
        lines.append(f"`{overall_winner}` is the best general-purpose imputer at "
                     f"{overall_median:.2f}% median rel. err — well within the range "
                     f"(< 5%) where downstream RUL training should benefit. "
                     "Apply to actual censored cells next.\n")
    elif overall_median < 10:
        lines.append(f"`{overall_winner}` ({overall_median:.2f}% median rel. err) "
                     "is acceptable but not great. Consider an ensemble of the top-3 "
                     "methods, or per-chemistry method selection.\n")
    else:
        lines.append(f"All methods exceed 10% median rel. err. Imputation is "
                     "noisy on this corpus regime. Recommend keeping censoring-stratified "
                     "reporting as the primary methodology.\n")

    (OUT_TBL / "findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
