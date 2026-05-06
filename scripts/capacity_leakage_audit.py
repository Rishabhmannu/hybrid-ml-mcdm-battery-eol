"""
Iteration-3 capacity-leakage audit.

SoH is *defined* as `capacity_Ah / nominal_Ah`. Our feature set currently
includes `capacity_Ah` plus 5 capacity-derived rolling/Δ columns. Predicting
SoH from those is therefore an arithmetic identity, not learning, and that
likely explains the headline R² ≈ 0.999 in Iter-1 / Iter-2.

This script trains two XGBoost SoH regressors back-to-back on the same
splits and reports the deltas:

  A) baseline           — current feature set (capacity-derived columns IN)
  B) leakage-audited    — capacity-derived columns DROPPED

If A's R² ≫ B's R², the headline number is largely tautological and the
project's framing should pivot to grade-routing accuracy. If they are
close, then the model is genuinely learning capacity from the
voltage/current/dQ/dV signals, and the headline stands.

Usage
-----
    python scripts/capacity_leakage_audit.py --smoke
    python scripts/capacity_leakage_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import (
    CAPACITY_LEAK_FEATURES,
    load_feature_bundle,
    soh_to_grade,
)
from src.utils.config import RANDOM_SEED, RESULTS_DIR, TARGETS, XGBOOST_CONFIG
from src.utils.metrics import stratified_regression_metrics

OUT_FIG = RESULTS_DIR / "figures" / "capacity_leakage_audit"
OUT_TBL = RESULTS_DIR / "tables" / "capacity_leakage_audit"


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _grade_accuracy(y_true_pct: np.ndarray, y_pred_pct: np.ndarray) -> dict:
    """Translate continuous SoH into A/B/C/D grades and report routing accuracy."""
    g_true = soh_to_grade(y_true_pct)
    g_pred = soh_to_grade(y_pred_pct)
    overall = float((g_true == g_pred).mean())
    per_grade = {}
    for g in ["A", "B", "C", "D"]:
        mask = g_true == g
        if mask.sum() == 0:
            continue
        per_grade[g] = {
            "n": int(mask.sum()),
            "accuracy": float((g_pred[mask] == g).mean()),
        }
    return {"overall": overall, "per_grade": per_grade}


def _train_one(name: str, exclude_capacity: bool, args) -> dict:
    print()
    print("─" * 70)
    print(f"[{name}]  exclude_capacity_features={exclude_capacity}")
    print("─" * 70)

    bundle = load_feature_bundle(
        smoke=args.smoke,
        exclude_capacity_features=exclude_capacity,
    )

    X_tr, y_tr = bundle.X_train, bundle.y_train_soh
    X_va, y_va = bundle.X_val, bundle.y_val_soh
    X_te, y_te = bundle.X_test, bundle.y_test_soh

    params = {**XGBOOST_CONFIG, "early_stopping_rounds": 30}
    if args.smoke:
        params["n_estimators"] = 50
        params["max_depth"] = 4

    t0 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    train_time = time.time() - t0

    pred_te = model.predict(X_te)
    test = _metrics(y_te, pred_te)
    grade = _grade_accuracy(y_te, pred_te)

    per_source = stratified_regression_metrics(y_te, pred_te, bundle.test_sources)
    per_chem = stratified_regression_metrics(y_te, pred_te, bundle.test_chemistries)

    print(f"  trained on {len(bundle.feature_names)} features in {train_time:.1f}s")
    print(f"  TEST  R²={test['r2']:+.4f}  RMSE={test['rmse']:.3f}%  MAE={test['mae']:.3f}%")
    print(f"  GRADE accuracy (A/B/C/D routing) = {grade['overall']*100:.2f}%")
    for g, info in grade["per_grade"].items():
        print(f"    {g}: n={info['n']:>7,d}  acc={info['accuracy']*100:5.2f}%")

    # Top-5 features for this run (sanity-check what the model is leaning on)
    top = (pd.DataFrame({"feature": bundle.feature_names,
                         "importance": model.feature_importances_})
           .sort_values("importance", ascending=False).head(5))
    print("  Top-5 features by gain:")
    for _, r in top.iterrows():
        print(f"    {r['feature']:35s}  {r['importance']:.4f}")

    return {
        "name": name,
        "exclude_capacity": exclude_capacity,
        "n_features": len(bundle.feature_names),
        "feature_names": bundle.feature_names,
        "feature_importances": model.feature_importances_.tolist(),
        "train_time_s": round(train_time, 2),
        "best_iteration": int(model.best_iteration),
        "test": test,
        "grade_accuracy": grade,
        "per_source": per_source,
        "per_chemistry": per_chem,
    }


def main():
    p = argparse.ArgumentParser(description="Iter-3 capacity-leakage audit")
    p.add_argument("--smoke", action="store_true",
                   help="Quick run on 25 k-row sample, 50 trees")
    args = p.parse_args()

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TBL.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Capacity-leakage audit  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    print(f"Dropping: {CAPACITY_LEAK_FEATURES}")

    np.random.seed(RANDOM_SEED)

    a = _train_one("A_baseline",   exclude_capacity=False, args=args)
    b = _train_one("B_audited",    exclude_capacity=True,  args=args)

    # ---- Side-by-side comparison ----
    suffix = "_smoke" if args.smoke else ""

    summary_rows = [
        {
            "config": cfg["name"],
            "n_features": cfg["n_features"],
            "test_r2": cfg["test"]["r2"],
            "test_rmse": cfg["test"]["rmse"],
            "test_mae": cfg["test"]["mae"],
            "grade_acc": cfg["grade_accuracy"]["overall"],
        }
        for cfg in (a, b)
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_TBL / f"summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    delta = {
        "delta_r2": a["test"]["r2"] - b["test"]["r2"],
        "delta_rmse": b["test"]["rmse"] - a["test"]["rmse"],
        "delta_mae": b["test"]["mae"] - a["test"]["mae"],
        "delta_grade_acc": a["grade_accuracy"]["overall"] - b["grade_accuracy"]["overall"],
    }
    print()
    print("=" * 70)
    print("LEAKAGE DELTA  (A_baseline − B_audited, where applicable)")
    print("=" * 70)
    print(f"  ΔR²            = {delta['delta_r2']:+.4f}    "
          f"(a={a['test']['r2']:.4f} → b={b['test']['r2']:.4f})")
    print(f"  ΔRMSE (b−a)    = {delta['delta_rmse']:+.3f}%   "
          f"(a={a['test']['rmse']:.3f} → b={b['test']['rmse']:.3f})")
    print(f"  ΔMAE  (b−a)    = {delta['delta_mae']:+.3f}%   "
          f"(a={a['test']['mae']:.3f} → b={b['test']['mae']:.3f})")
    print(f"  Δgrade-acc     = {delta['delta_grade_acc']*100:+.2f}pp  "
          f"(a={a['grade_accuracy']['overall']*100:.2f}% "
          f"→ b={b['grade_accuracy']['overall']*100:.2f}%)")

    payload = {
        "smoke": args.smoke,
        "dropped_features": CAPACITY_LEAK_FEATURES,
        "A_baseline": {**{k: v for k, v in a.items() if k not in
                          ("feature_names", "feature_importances", "per_source", "per_chemistry")}},
        "B_audited":  {**{k: v for k, v in b.items() if k not in
                          ("feature_names", "feature_importances", "per_source", "per_chemistry")}},
        "delta": delta,
    }
    json_path = OUT_TBL / f"results{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    pd.DataFrame(a["per_source"]).to_csv(OUT_TBL / f"A_per_source{suffix}.csv", index=False)
    pd.DataFrame(b["per_source"]).to_csv(OUT_TBL / f"B_per_source{suffix}.csv", index=False)
    pd.DataFrame(a["per_chemistry"]).to_csv(OUT_TBL / f"A_per_chemistry{suffix}.csv", index=False)
    pd.DataFrame(b["per_chemistry"]).to_csv(OUT_TBL / f"B_per_chemistry{suffix}.csv", index=False)

    # ---- Findings markdown ----
    verdict_lines = []
    if delta["delta_r2"] > 0.10:
        verdict_lines.append(
            f"**Strong leakage signal.** ΔR² = {delta['delta_r2']:+.4f} indicates the "
            "headline R² is largely an arithmetic identity (capacity → SoH = capacity / "
            "nominal). The audited number is the one to report."
        )
    elif delta["delta_r2"] > 0.02:
        verdict_lines.append(
            f"**Moderate leakage.** ΔR² = {delta['delta_r2']:+.4f}. Capacity features add "
            "real signal but they are not the only thing the model relies on; the audited "
            "model is still informative."
        )
    else:
        verdict_lines.append(
            f"**No meaningful leakage.** ΔR² = {delta['delta_r2']:+.4f}. The model learns "
            "SoH primarily from voltage/current/dQ/dV signals; capacity features are "
            "secondary."
        )

    md_path = OUT_TBL / f"findings{suffix}.md"
    md_path.write_text("\n".join([
        "# Capacity-leakage audit (XGBoost SoH)",
        "",
        f"_Mode: {'SMOKE' if args.smoke else 'FULL'}  ·  dropped features: "
        f"{CAPACITY_LEAK_FEATURES}_",
        "",
        "## Headline",
        "",
        verdict_lines[0],
        "",
        "## Comparison",
        "",
        "| Config | n features | Test R² | Test RMSE | Test MAE | Grade-routing acc |",
        "|---|---|---|---|---|---|",
        f"| A_baseline (capacity IN) | {a['n_features']} | {a['test']['r2']:+.4f} | "
        f"{a['test']['rmse']:.3f}% | {a['test']['mae']:.3f}% | "
        f"{a['grade_accuracy']['overall']*100:.2f}% |",
        f"| B_audited (capacity OUT) | {b['n_features']} | {b['test']['r2']:+.4f} | "
        f"{b['test']['rmse']:.3f}% | {b['test']['mae']:.3f}% | "
        f"{b['grade_accuracy']['overall']*100:.2f}% |",
        f"| **Δ (A − B)** | — | **{delta['delta_r2']:+.4f}** | "
        f"**{-delta['delta_rmse']:+.3f}%** | **{-delta['delta_mae']:+.3f}%** | "
        f"**{delta['delta_grade_acc']*100:+.2f}pp** |",
        "",
        "## Interpretation",
        "",
        "- SoH is **defined** as `capacity_Ah / nominal_Ah`. Including `capacity_Ah` "
        "(or its rolling means / Δ) as a feature lets the model recover the target "
        "via division — it's an arithmetic identity, not generalization.",
        "- The **audited** column (B) is the realistic deployment scenario: at "
        "second-life triage, capacity is what we are *trying to estimate*, not an input.",
        "- Grade-routing accuracy (A/B/C/D) is the project's actual decision metric — "
        "downstream MCDM allocates retired cells into reuse buckets, not into "
        "fractional SoH percentiles. A 2 % SoH RMSE only matters if it changes the bucket.",
        "",
        "## What to report in the paper",
        "",
        "- Lead with **B_audited's R²**, RMSE, MAE, and grade-routing accuracy.",
        "- Disclose **A_baseline** as the literature-style number to make the "
        "leakage explicit and defend the methodology.",
        "- The delta itself is a contribution: most published SoH-from-capacity "
        "studies do not run this audit.",
    ]) + "\n")

    print()
    print(f"summary  → {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"json     → {json_path.relative_to(PROJECT_ROOT)}")
    print(f"findings → {md_path.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
