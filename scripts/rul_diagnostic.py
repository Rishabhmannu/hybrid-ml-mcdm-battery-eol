"""
RUL right-censoring diagnostic — investigate Hypothesis A.

Quantifies what fraction of batteries in the unified corpus actually reached
End-of-Life (SoH < 0.8) in the experimental data, vs. were stopped before
reaching it ("right-censored" — their TRUE RUL is unknown, but our training
pipeline currently fabricates a label using `max(observed_cycle) - current_cycle`).

Right-censored cells contribute systematically biased labels: the model is
told they have less remaining life than they truly do. If the censoring rate
is high (> 30 %), this is the dominant source of error in the XGBoost-RUL
test RMSE that's currently failing the strict 2 % gate.

This script does NOT retrain anything. It:

  1. Computes per-battery censoring stats from unified.parquet
  2. Reports right-censoring rate overall + per source + per chemistry + per split
  3. Loads the existing audited XGBoost RUL model
  4. Computes test residuals stratified by censoring status of the source battery
  5. Estimates what test RMSE would be on UNCENSORED-only test cells
     (the "true-EoL" subset where labels are physically meaningful)

Outputs:
  results/tables/rul_diagnostic/
    overall_stats.json
    per_source.csv
    per_chemistry.csv
    per_split.csv
    residual_by_censoring.json
    findings.md
  results/figures/rul_diagnostic/
    censoring_per_source.png
    soh_min_distribution.png
    rul_distribution.png
    residual_by_censoring.png

Usage
-----
    python scripts/rul_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import EOL_THRESHOLD, load_feature_bundle
from src.utils.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR

UNIFIED_PARQUET = PROCESSED_DIR / "cycling" / "unified.parquet"
SPLITS_JSON = PROCESSED_DIR / "cycling" / "splits.json"
MODEL_PATH = MODELS_DIR / "xgboost_rul" / "xgboost_rul_audited.json"

OUT_TBL = RESULTS_DIR / "tables" / "rul_diagnostic"
OUT_FIG = RESULTS_DIR / "figures" / "rul_diagnostic"


# =============================================================================
# Section 1: per-battery censoring stats
# =============================================================================

def compute_battery_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per battery — n_cycles, min/max SoH, censoring status, etc."""
    g = df.groupby("battery_id", sort=False)
    stats = pd.DataFrame({
        "n_cycles":   g["cycle"].count(),
        "max_cycle":  g["cycle"].max(),
        "min_soh":    g["soh"].min(),
        "max_soh":    g["soh"].max(),
        "first_soh":  g["soh"].first(),
        "last_soh":   g["soh"].last(),
        "chemistry":  g["chemistry"].first(),
        "source":     g["source"].first(),
    })
    stats["crossed_eol"] = stats["min_soh"] < EOL_THRESHOLD
    stats["right_censored"] = ~stats["crossed_eol"]

    # How close was a censored cell to EoL? Distance = min_soh - 0.8.
    # 0 means basically at EoL; 0.2 means cell stopped at SoH=1.0.
    stats["distance_to_eol"] = stats["min_soh"] - EOL_THRESHOLD

    return stats.reset_index()


# =============================================================================
# Section 2: aggregate stats helpers
# =============================================================================

def per_group_summary(stats: pd.DataFrame, by: str) -> pd.DataFrame:
    g = stats.groupby(by)
    out = pd.DataFrame({
        "n":              g.size(),
        "n_uncensored":   g["crossed_eol"].sum(),
        "n_censored":     g["right_censored"].sum(),
        "median_min_soh": g["min_soh"].median(),
        "mean_min_soh":   g["min_soh"].mean(),
        "median_n_cycles": g["n_cycles"].median(),
    })
    out["censoring_rate"] = out["n_censored"] / out["n"]
    out = out.sort_values("n", ascending=False).reset_index()
    return out


def per_split_summary(stats: pd.DataFrame, splits: dict) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "val", "test"):
        bids = splits.get(split_name, [])
        sub = stats[stats["battery_id"].isin(bids)]
        rows.append({
            "split": split_name,
            "n_batteries": len(sub),
            "n_uncensored": int(sub["crossed_eol"].sum()),
            "n_censored":   int(sub["right_censored"].sum()),
            "censoring_rate": float(sub["right_censored"].mean()),
            "median_min_soh": float(sub["min_soh"].median()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Section 3: residual analysis on the existing audited model
# =============================================================================

def residuals_by_censoring(stats: pd.DataFrame) -> dict:
    """Load audited XGBoost RUL model, predict on test, stratify residuals
    by whether the source battery is right-censored."""

    if not MODEL_PATH.exists():
        print(f"[WARN] {MODEL_PATH} not found — skipping residual analysis.")
        return {}

    print(f"\n[Residuals] Loading bundle (audited mode) and model ...")
    bundle = load_feature_bundle(exclude_capacity_features=True, verbose=False)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    pred_te = model.predict(bundle.X_test)
    y_te = bundle.y_test_rul
    test_battery_ids = bundle.test_battery_ids.values

    censor_map = stats.set_index("battery_id")["right_censored"].to_dict()
    test_censored = np.array([censor_map.get(bid, False) for bid in test_battery_ids])

    target_range = float(np.ptp(bundle.y_train_rul)) if len(bundle.y_train_rul) else 1.0

    def _metrics(mask, label):
        if mask.sum() == 0:
            return {"label": label, "n": 0}
        y, p = y_te[mask], pred_te[mask]
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        mae = float(mean_absolute_error(y, p))
        return {
            "label": label,
            "n": int(mask.sum()),
            "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "rmse_cycles": rmse,
            "mae_cycles": mae,
            "rmse_pct_of_range": rmse / max(target_range, 1.0) * 100.0,
            "mae_pct_of_range": mae / max(target_range, 1.0) * 100.0,
        }

    overall = _metrics(np.ones_like(test_censored, dtype=bool), "overall")
    uncensored = _metrics(~test_censored, "uncensored_only")
    censored = _metrics(test_censored, "censored_only")

    out = {
        "target_range_cycles": target_range,
        "overall": overall,
        "uncensored_only": uncensored,
        "censored_only": censored,
    }

    print("\n[Residuals] Test metrics stratified by censoring status:")
    for k in ["overall", "uncensored_only", "censored_only"]:
        m = out[k]
        if m["n"] == 0:
            continue
        print(f"  {k:18s}  n={m['n']:>7,d}  R²={m.get('r2', float('nan')):.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")

    # Plot: residual distribution censored vs uncensored
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    res = pred_te - y_te
    ax[0].hist(res[~test_censored], bins=80, alpha=0.6, label=f"uncensored (n={(~test_censored).sum():,})", density=True)
    ax[0].hist(res[test_censored], bins=80, alpha=0.6, label=f"censored (n={test_censored.sum():,})", density=True)
    ax[0].set_xlabel("Residual (predicted − true RUL, cycles)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Residual distribution by censoring status")
    ax[0].legend()
    ax[0].axvline(0, color="black", linewidth=0.5, linestyle="--")

    # Predicted vs true scatter, color-coded
    samp = np.random.RandomState(42).choice(len(y_te), size=min(20000, len(y_te)), replace=False)
    ax[1].scatter(y_te[samp][~test_censored[samp]], pred_te[samp][~test_censored[samp]],
                  s=2, alpha=0.4, label="uncensored")
    ax[1].scatter(y_te[samp][test_censored[samp]], pred_te[samp][test_censored[samp]],
                  s=2, alpha=0.4, label="censored")
    lim = max(y_te.max(), pred_te.max())
    ax[1].plot([0, lim], [0, lim], "k--", linewidth=0.5)
    ax[1].set_xlabel("True RUL (cycles)")
    ax[1].set_ylabel("Predicted RUL (cycles)")
    ax[1].set_title("Predicted vs true (color = censoring status)")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG / "residual_by_censoring.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {(OUT_FIG / 'residual_by_censoring.png').relative_to(PROJECT_ROOT)}")
    return out


# =============================================================================
# Section 4: figures
# =============================================================================

def plot_soh_min_distribution(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stats["min_soh"], bins=80, alpha=0.8)
    ax.axvline(EOL_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"EoL threshold (SoH={EOL_THRESHOLD})")
    ax.set_xlabel("Minimum SoH observed per battery")
    ax.set_ylabel("Number of batteries")
    ax.set_title(f"Right-censoring diagnostic: cells right of red line never reached EoL "
                 f"(n={stats['right_censored'].sum()}/{len(stats)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "soh_min_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {(OUT_FIG / 'soh_min_distribution.png').relative_to(PROJECT_ROOT)}")


def plot_censoring_per_source(per_source: pd.DataFrame, top_n: int = 25):
    sub = per_source.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(sub))
    ax.barh(y, sub["censoring_rate"], alpha=0.85)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(r["censoring_rate"] + 0.005, i,
                f"{r['n_censored']:,}/{r['n']:,} (med min-SoH {r['median_min_soh']:.2f})",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["source"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Censoring rate (fraction of batteries that did NOT reach SoH < 0.8)")
    ax.set_title(f"Right-censoring rate per source (top {top_n} by battery count)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "censoring_per_source.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {(OUT_FIG / 'censoring_per_source.png').relative_to(PROJECT_ROOT)}")


def plot_rul_distribution(df_rul: pd.DataFrame, stats: pd.DataFrame):
    censor_map = stats.set_index("battery_id")["right_censored"].to_dict()
    df_rul = df_rul.copy()
    df_rul["right_censored"] = df_rul["battery_id"].map(censor_map)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(df_rul.loc[~df_rul["right_censored"], "rul"], bins=120, alpha=0.6,
               label=f"uncensored cycles", density=True)
    ax[0].hist(df_rul.loc[df_rul["right_censored"], "rul"], bins=120, alpha=0.6,
               label=f"censored cycles", density=True)
    ax[0].set_xlabel("RUL target (cycles)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("RUL target distribution by censoring status")
    ax[0].legend()

    # Log-space view (Hypothesis B preview)
    ax[1].hist(np.log1p(df_rul.loc[~df_rul["right_censored"], "rul"]), bins=120, alpha=0.6,
               label="uncensored", density=True)
    ax[1].hist(np.log1p(df_rul.loc[df_rul["right_censored"], "rul"]), bins=120, alpha=0.6,
               label="censored", density=True)
    ax[1].set_xlabel("log(1 + RUL)")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Log-RUL distribution (preview for Hypothesis B)")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG / "rul_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {(OUT_FIG / 'rul_distribution.png').relative_to(PROJECT_ROOT)}")


# =============================================================================
# Section 5: orchestrator
# =============================================================================

def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUL right-censoring diagnostic")
    print("=" * 70)
    print(f"Loading {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    print(f"  {len(df):,} rows · {df['battery_id'].nunique():,} batteries · "
          f"{df['source'].nunique()} sources · {df['chemistry'].nunique()} chemistries")

    # ---- Section 1: battery-level stats -----------------------------------
    stats = compute_battery_stats(df)

    n_total = len(stats)
    n_uncensored = int(stats["crossed_eol"].sum())
    n_censored = int(stats["right_censored"].sum())
    rate = n_censored / max(n_total, 1)

    overall = {
        "n_batteries": n_total,
        "n_uncensored": n_uncensored,
        "n_censored": n_censored,
        "censoring_rate": rate,
        "median_min_soh_overall": float(stats["min_soh"].median()),
        "median_min_soh_censored": float(stats.loc[stats["right_censored"], "min_soh"].median()),
        "median_min_soh_uncensored": float(stats.loc[stats["crossed_eol"], "min_soh"].median()),
        "eol_threshold": EOL_THRESHOLD,
    }
    print(f"\n[Overall]")
    print(f"  Total batteries:      {n_total:,}")
    print(f"  Reached EoL (SoH<0.8): {n_uncensored:,}  ({(1-rate)*100:.1f}%)")
    print(f"  Right-censored:        {n_censored:,}  ({rate*100:.1f}%)")
    print(f"  Median min-SoH (uncensored): {overall['median_min_soh_uncensored']:.3f}")
    print(f"  Median min-SoH (censored):   {overall['median_min_soh_censored']:.3f}")

    with open(OUT_TBL / "overall_stats.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"[Save] {(OUT_TBL / 'overall_stats.json').relative_to(PROJECT_ROOT)}")

    # ---- Section 2: per source / per chemistry / per split ----------------
    per_source = per_group_summary(stats, "source")
    per_chemistry = per_group_summary(stats, "chemistry")
    per_split = per_split_summary(stats, splits)

    print(f"\n[Per source — top 15 by battery count]")
    for _, r in per_source.head(15).iterrows():
        bar = "█" * int(r["censoring_rate"] * 30)
        print(f"  {r['source']:35s}  n={int(r['n']):>4d}  "
              f"censored={r['censoring_rate']*100:5.1f}%  {bar}")
    print(f"\n[Per chemistry]")
    for _, r in per_chemistry.iterrows():
        print(f"  {r['chemistry']:10s}  n={int(r['n']):>5d}  "
              f"censored={r['censoring_rate']*100:5.1f}%  "
              f"med_min_soh={r['median_min_soh']:.3f}")
    print(f"\n[Per split]")
    for _, r in per_split.iterrows():
        print(f"  {r['split']:6s}  n={int(r['n_batteries']):>4d}  "
              f"censored={r['censoring_rate']*100:5.1f}%  "
              f"med_min_soh={r['median_min_soh']:.3f}")

    per_source.to_csv(OUT_TBL / "per_source.csv", index=False)
    per_chemistry.to_csv(OUT_TBL / "per_chemistry.csv", index=False)
    per_split.to_csv(OUT_TBL / "per_split.csv", index=False)
    print(f"[Save] per_source.csv · per_chemistry.csv · per_split.csv")

    # ---- Section 3: figures ------------------------------------------------
    print(f"\n[Figures]")
    plot_soh_min_distribution(stats)
    plot_censoring_per_source(per_source)

    # Build RUL frame for distribution plot (matches training pipeline logic)
    from src.data.training_data import _build_feature_frame, NUMERIC_FEATURES
    df_rul = _build_feature_frame(df, NUMERIC_FEATURES)
    plot_rul_distribution(df_rul[["battery_id", "rul"]].copy(), stats)

    # ---- Section 4: residuals by censoring on the audited model -----------
    residual_stats = residuals_by_censoring(stats)
    if residual_stats:
        with open(OUT_TBL / "residual_by_censoring.json", "w") as f:
            json.dump(residual_stats, f, indent=2)
        print(f"[Save] {(OUT_TBL / 'residual_by_censoring.json').relative_to(PROJECT_ROOT)}")

    # ---- Section 5: findings.md -------------------------------------------
    findings = build_findings(overall, per_source, per_chemistry, per_split, residual_stats)
    (OUT_TBL / "findings.md").write_text(findings)
    print(f"[Save] {(OUT_TBL / 'findings.md').relative_to(PROJECT_ROOT)}")
    print("\nDone.")


def build_findings(overall, per_source, per_chemistry, per_split, residual_stats) -> str:
    rate = overall["censoring_rate"]

    if rate < 0.15:
        verdict = ("**LOW censoring rate** — Hypothesis A is unlikely to be the dominant "
                   "issue. Move to Hypotheses B (log-target / Tweedie loss) or D "
                   "(per-chemistry submodels) for the next experiment.")
    elif rate < 0.40:
        verdict = ("**MODERATE censoring rate** — Hypothesis A is meaningful but probably "
                   "not the only cause. Try the survival-regression intervention AND "
                   "Hypothesis B/D in parallel.")
    else:
        verdict = ("**HIGH censoring rate** — Hypothesis A is almost certainly the "
                   "dominant issue. Survival regression (XGBoost objective='survival:aft') "
                   "or excluding censored cells from training is the priority intervention.")

    # Compare per-censoring-status RMSE if model was loaded.
    residual_block = ""
    if residual_stats:
        u = residual_stats.get("uncensored_only", {})
        c = residual_stats.get("censored_only", {})
        if u and c and u.get("n", 0) > 0 and c.get("n", 0) > 0:
            ratio = c.get("rmse_cycles", 0) / max(u.get("rmse_cycles", 1), 1)
            residual_block = f"""
## Residual analysis on the audited XGBoost-RUL model

| Subset | n | RMSE (cyc) | RMSE (% of range) | MAE (cyc) | R² |
|---|---|---|---|---|---|
| Overall | {residual_stats['overall']['n']:,} | {residual_stats['overall']['rmse_cycles']:.1f} | {residual_stats['overall']['rmse_pct_of_range']:.2f}% | {residual_stats['overall']['mae_cycles']:.1f} | {residual_stats['overall'].get('r2', float('nan')):.4f} |
| Uncensored only | {u['n']:,} | {u['rmse_cycles']:.1f} | {u['rmse_pct_of_range']:.2f}% | {u['mae_cycles']:.1f} | {u.get('r2', float('nan')):.4f} |
| Censored only | {c['n']:,} | {c['rmse_cycles']:.1f} | {c['rmse_pct_of_range']:.2f}% | {c['mae_cycles']:.1f} | {c.get('r2', float('nan')):.4f} |

Censored-cell RMSE is **{ratio:.2f}× the uncensored RMSE**.
"""
            if u.get("rmse_pct_of_range", 100) < 2.0:
                residual_block += ("\n**Implication: on uncensored test cells alone the "
                                   "audited XGBoost-RUL ALREADY PASSES the 2 % gate.** The "
                                   "headline test RMSE is being dragged up by fabricated "
                                   "labels on right-censored cells — a label problem, not "
                                   "a model problem.")
            elif ratio > 1.5:
                residual_block += ("\n**Implication: censored-cell residuals dominate the "
                                   "headline RMSE.** Fixing the labels (or excluding them) "
                                   "should close most of the gap to the 2 % gate.")

    return f"""# RUL right-censoring diagnostic — findings

_Generated by `scripts/rul_diagnostic.py`. EoL threshold = SoH < {overall['eol_threshold']}._

## Headline numbers

- **{overall['n_batteries']:,} batteries in unified corpus.**
- **{overall['n_uncensored']:,} ({(1-rate)*100:.1f} %) reached EoL** in the experimental data.
- **{overall['n_censored']:,} ({rate*100:.1f} %) are right-censored** — they never reached SoH < {overall['eol_threshold']} in the data, so their RUL labels are fabricated as `max(observed_cycle) − current_cycle`.
- Median min-SoH for censored cells: **{overall['median_min_soh_censored']:.3f}** (so on average a censored cell stopped {(overall['median_min_soh_censored'] - overall['eol_threshold']) * 100:.1f} pp above the EoL threshold).

{verdict}
{residual_block}

## Per-split censoring rate (anti-leakage check)

| Split | n batteries | uncensored | censored | rate |
|---|---|---|---|---|
{chr(10).join(f"| {r['split']} | {int(r['n_batteries'])} | {int(r['n_uncensored'])} | {int(r['n_censored'])} | {r['censoring_rate']*100:.1f}% |" for _, r in per_split.iterrows())}

A roughly equal censoring rate across splits means the issue is corpus-wide, not split-specific.

## Top-10 sources by censoring rate (among sources with n ≥ 10)

| Source | n | censored | rate | median min-SoH |
|---|---|---|---|---|
{chr(10).join(f"| {r['source']} | {int(r['n'])} | {int(r['n_censored'])} | {r['censoring_rate']*100:.1f}% | {r['median_min_soh']:.3f} |" for _, r in per_source[per_source['n'] >= 10].sort_values('censoring_rate', ascending=False).head(10).iterrows())}

## Per chemistry

| Chemistry | n | censored | rate | median min-SoH |
|---|---|---|---|---|
{chr(10).join(f"| {r['chemistry']} | {int(r['n'])} | {int(r['n_censored'])} | {r['censoring_rate']*100:.1f}% | {r['median_min_soh']:.3f} |" for _, r in per_chemistry.iterrows())}

## Next-step decision tree

1. **If censoring rate > 30 % AND censored-cell RMSE >> uncensored-cell RMSE** → Hypothesis A confirmed dominant. Try:
   a. Train XGBoost RUL with `objective="survival:aft"` (proper handling of right-censored data)
   b. Train XGBoost RUL excluding right-censored cells from training (cleaner signal, less data)
   c. Compare both against the current baseline.

2. **If censoring rate is high but censored-cell RMSE ≈ uncensored-cell RMSE** → labels aren't the issue; the model is generalizing similarly across both. Move to Hypothesis B (log-target) and D (per-chemistry router).

3. **If censoring rate < 15 %** → Hypothesis A is not the issue. Move directly to Hypotheses B/D.
"""


if __name__ == "__main__":
    main()
