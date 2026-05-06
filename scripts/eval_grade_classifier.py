"""
Stage 9 — Evaluate the SoH→Grade classifier.

Loads the trained XGBoost SoH model, derives A/B/C/D grades from its predictions,
compares against ground-truth grades from the test set. Reports accuracy / F1 +
confusion matrix.

Usage
-----
    python scripts/eval_grade_classifier.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import load_feature_bundle, soh_to_grade
from src.utils.config import MODELS_DIR, RESULTS_DIR, TARGETS
from src.utils.metrics import stratified_classification_metrics
from src.utils.plots import plot_confusion_matrix

OUT_DIR = MODELS_DIR / "grade_classifier"
FIG_DIR = RESULTS_DIR / "figures" / "grade_classifier"
TBL_DIR = RESULTS_DIR / "tables" / "grade_classifier"


def main():
    p = argparse.ArgumentParser(description="Evaluate SoH→Grade classifier")
    p.add_argument("--smoke", action="store_true", help="Use the smoke feature bundle")
    p.add_argument("--xgb-model", default=None,
                   help="Path to XGBoost SoH model. Auto-resolved from --exclude-capacity "
                        "if not provided.")
    p.add_argument("--exclude-capacity", action="store_true",
                   help="Iter-3 audited mode: load the audited (capacity-excluded) model "
                        "and a matching feature bundle.")
    args = p.parse_args()
    suffix = "_audited" if args.exclude_capacity else ""
    if args.xgb_model is None:
        args.xgb_model = str(MODELS_DIR / "xgboost_soh" / f"xgboost_soh{suffix}.json")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    xgb_path = Path(args.xgb_model)
    if not xgb_path.exists():
        raise SystemExit(f"XGBoost model not found at {xgb_path}. Run train_xgboost_soh.py first.")

    print("=" * 70)
    print(f"Grade classifier evaluation  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)

    bundle = load_feature_bundle(smoke=args.smoke,
                                 exclude_capacity_features=args.exclude_capacity)
    model = xgb.XGBRegressor()
    model.load_model(str(xgb_path))

    soh_pred_val = model.predict(bundle.X_val)
    soh_pred_test = model.predict(bundle.X_test)

    grade_true_val = soh_to_grade(bundle.y_val_soh)
    grade_pred_val = soh_to_grade(soh_pred_val)
    grade_true_test = soh_to_grade(bundle.y_test_soh)
    grade_pred_test = soh_to_grade(soh_pred_test)

    labels = ["A", "B", "C", "D"]

    def _eval(y_true, y_pred, name: str):
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        f1p = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        print(f"\n[{name}]")
        print(f"  accuracy = {acc:.4f}")
        print(f"  F1 macro = {f1m:.4f}")
        print(f"  F1 per class: " + ", ".join(f"{l}={p:.3f}" for l, p in zip(labels, f1p)))
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        return {"accuracy": acc, "f1_macro": f1m,
                "f1_per_class": dict(zip(labels, [float(x) for x in f1p]))}

    val_metrics  = _eval(grade_true_val,  grade_pred_val,  "VAL")
    test_metrics = _eval(grade_true_test, grade_pred_test, "TEST")

    # ---- Stratified eval per Stage-EDA recommendation ----
    per_source = stratified_classification_metrics(
        grade_true_test, grade_pred_test, bundle.test_sources, labels=labels)
    per_chem = stratified_classification_metrics(
        grade_true_test, grade_pred_test, bundle.test_chemistries, labels=labels)
    print("\n[TEST] Per-source accuracy + F1 macro (top 10):")
    for r in per_source[:10]:
        if pd.notna(r["accuracy"]):
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  "
                  f"acc={r['accuracy']:.4f}  F1macro={r['f1_macro']:.4f}")
        else:
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  (below min_n)")
    print("\n[TEST] Per-chemistry accuracy + F1 macro:")
    for r in per_chem:
        if pd.notna(r["accuracy"]):
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  "
                  f"acc={r['accuracy']:.4f}  F1macro={r['f1_macro']:.4f}")
        else:
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  (below min_n)")
    pd.DataFrame(per_source).to_csv(TBL_DIR / f"test_grade_metrics_per_source{suffix}.csv", index=False)
    pd.DataFrame(per_chem).to_csv(TBL_DIR / f"test_grade_metrics_per_chemistry{suffix}.csv", index=False)

    gates = {
        "accuracy ≥ 0.90": test_metrics["accuracy"] >= TARGETS["grade_accuracy"],
        "f1_macro ≥ 0.85": test_metrics["f1_macro"] >= TARGETS["grade_f1"],
    }
    print("\n[Gates] " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()))

    title_tag = " (audited)" if args.exclude_capacity else ""
    plot_confusion_matrix(
        grade_true_test, grade_pred_test, labels=labels,
        out_path=FIG_DIR / f"confusion_matrix_test{suffix}.png",
        title=f"Grade classifier{title_tag} — Test confusion matrix")
    plot_confusion_matrix(
        grade_true_test, grade_pred_test, labels=labels,
        out_path=FIG_DIR / f"confusion_matrix_test_normalized{suffix}.png",
        title=f"Grade classifier{title_tag} — Test (row-normalized)",
        normalize=True)

    metrics = {
        "smoke": args.smoke,
        "xgb_model": str(xgb_path),
        "val": val_metrics,
        "test": test_metrics,
        "gates": gates,
    }
    with open(TBL_DIR / f"metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-grade SoH prediction quality
    df = pd.DataFrame({
        "soh_true": bundle.y_test_soh,
        "soh_pred": soh_pred_test,
        "grade_true": grade_true_test,
        "grade_pred": grade_pred_test,
    })
    df.to_csv(TBL_DIR / f"test_predictions{suffix}.csv", index=False)
    print(f"\n[Save] metrics → {(TBL_DIR / f'metrics{suffix}.json').relative_to(PROJECT_ROOT)}")
    print(f"[Save] preds   → {(TBL_DIR / f'test_predictions{suffix}.csv').relative_to(PROJECT_ROOT)}")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
