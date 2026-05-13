"""
Iter-3 §3.11.5 — Train XGBoost RUL regressor on unified.parquet.

Replaces the LSTM RUL model (Iter-1/2/3 all failed the strict 2 % RMSE-of-range
gate; Iter-3 best was 4.89 % with a 0.14 train→test R² gap — architectural
overfit, not a data issue). Per Severson et al. 2019 (Nature Energy 4),
Roman et al. 2021 (Nat. Mach. Intel. 3), and Tian et al. 2023 (Energy 277),
gradient-boosted trees on per-cycle summary features match or exceed sequence
models for battery aging tasks on heterogeneous medium-scale corpora.

Reuses the audited feature pipeline from `load_feature_bundle(...)` so
XGBoost RUL is a clean head-to-head against the LSTM (same train/val/test
splits, same numeric features, same anti-leakage rules). Target swapped from
`soh_pct` to `rul` (cycles-to-EoL @ SoH=0.8, computed in
`src.data.training_data._compute_rul_per_battery`).

Target per [src/utils/config.py::TARGETS]: RMSE < 2 % of train RUL range,
mirroring the LSTM gate.

Usage
-----
    python scripts/train_xgboost_rul.py --smoke                   # 60 sec sanity run
    python scripts/train_xgboost_rul.py                           # full training (default mode)
    python scripts/train_xgboost_rul.py --exclude-capacity        # audited mode (matches SoH audit)
    python scripts/train_xgboost_rul.py --tune --trials 50        # full + Optuna
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import compute_battery_censoring, load_feature_bundle
from src.utils.config import MODELS_DIR, RESULTS_DIR, TARGETS, XGBOOST_CONFIG, RANDOM_SEED
from src.utils.metrics import stratified_regression_metrics
from src.utils.plots import (
    plot_feature_importance,
    plot_loss_curves,
    plot_overfit_check,
    plot_predicted_vs_actual,
    plot_residuals,
)

OUT_DIR = MODELS_DIR / "xgboost_rul"
FIG_DIR = RESULTS_DIR / "figures" / "xgboost_rul"
TBL_DIR = RESULTS_DIR / "tables" / "xgboost_rul"


def _metrics(y_true, y_pred, target_range: float) -> dict:
    """Mirror src.utils metrics shape used by LSTM RUL so the head-to-head
    comparison is apples-to-apples."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_cycles": rmse,
        "mae_cycles": mae,
        "rmse_pct_of_range": rmse / max(target_range, 1.0) * 100.0,
        "mae_pct_of_range": mae / max(target_range, 1.0) * 100.0,
    }


def _run_optuna(X_train, y_train, X_val, y_val, n_trials: int) -> dict:
    import optuna  # local import keeps `--smoke` runs fast
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[Optuna] best R² = {study.best_value:.4f}")
    print(f"[Optuna] best params: {study.best_params}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, OUT_DIR / "optuna_study.pkl")
    return study.best_params


def _maybe_run_shap(model, X_test, feature_names, n_samples: int, suffix: str = ""):
    """SHAP is slow on full test set — sample down for plots."""
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed, skipping")
        return
    n = min(n_samples, len(X_test))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X_test), n, replace=False)
    X_sample = pd.DataFrame(X_test[idx], columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    import matplotlib.pyplot as plt
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_bar{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_beeswarm{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False))
    top.to_csv(TBL_DIR / f"shap_top_features{suffix}.csv", index=False)
    print(f"[SHAP] top 5 by mean |SHAP|:")
    print(top.head().to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Train XGBoost RUL regressor")
    p.add_argument("--smoke", action="store_true", help="Smoke run: tiny sample, few trees")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials")
    p.add_argument("--shap-samples", type=int, default=2000, help="SHAP sample size")
    p.add_argument("--include-high-missing", action="store_true",
                   help="Include ir_ohm + temperature features (87-93%% missing — for ablation only)")
    p.add_argument("--exclude-capacity", action="store_true",
                   help="Iter-3 audited mode: drop capacity_Ah and 5 capacity-derived "
                        "rolling/Δ features. Matches the audited SoH model's feature set "
                        "for the head-to-head LSTM-vs-XGBoost RUL comparison.")
    p.add_argument("--exclude-censored", action="store_true",
                   help="Iter-3 §3.11.5 Hypothesis A intervention: drop right-censored "
                        "batteries (never reached SoH<0.8) from TRAIN and VAL only. "
                        "Test stays full so we can report both headline and uncensored-only "
                        "test metrics. Cleaner labels at the cost of ~19% less training data.")
    p.add_argument("--use-imputed-rul", action="store_true",
                   help="Iter-3 §3.11.5 Hypothesis A intervention (preferred): replace "
                        "fabricated RUL labels for right-censored cells with imputed labels "
                        "from data/processed/cycling/imputed_rul_labels.csv "
                        "(produced by scripts/apply_rul_imputation.py). Per held-out "
                        "validation: imputation median rel err ~1.5% vs ~16-19% implicit "
                        "error from `max-cycle` fallback. Mutually exclusive with "
                        "--exclude-censored (use one or the other).")
    args = p.parse_args()
    if args.use_imputed_rul and args.exclude_censored:
        p.error("--use-imputed-rul and --exclude-censored are mutually exclusive — "
                "use one approach to handle right-censoring, not both.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"XGBoost RUL training  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    bundle = load_feature_bundle(smoke=args.smoke,
                                 include_high_missing=args.include_high_missing,
                                 exclude_capacity_features=args.exclude_capacity,
                                 exclude_censored_batteries=args.exclude_censored,
                                 use_imputed_rul=args.use_imputed_rul)
    suffix_parts = []
    if args.exclude_capacity:
        suffix_parts.append("audited")
    if args.exclude_censored:
        suffix_parts.append("uncensored")
    if args.use_imputed_rul:
        suffix_parts.append("imputed")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    X_tr, y_tr = bundle.X_train, bundle.y_train_rul
    X_va, y_va = bundle.X_val,   bundle.y_val_rul
    X_te, y_te = bundle.X_test,  bundle.y_test_rul

    # Train RUL range — same denominator the LSTM script uses for
    # `rmse_pct_of_range`. Computed on train only so val/test rescaling stays
    # honest under sample shifts.
    target_range = float(np.ptp(y_tr)) if len(y_tr) > 0 else 1.0
    print(f"\n[Data] RUL range (train) = {target_range:.0f} cycles  "
          f"(min={y_tr.min():.0f}, max={y_tr.max():.0f})")
    print(f"[Data] n_train={len(y_tr):,}  n_val={len(y_va):,}  n_test={len(y_te):,}  "
          f"n_features={len(bundle.feature_names)}")

    if args.tune and not args.smoke:
        params = _run_optuna(X_tr, y_tr, X_va, y_va, n_trials=args.trials)
    else:
        params = {**XGBOOST_CONFIG}

    if args.smoke:
        params["n_estimators"] = 50
        params["max_depth"] = 4

    params["early_stopping_rounds"] = 30
    print(f"\n[Train] params: {json.dumps(params, default=str, indent=2)}")

    t0 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=False,
    )
    train_time = time.time() - t0

    eval_results = model.evals_result()
    rmse_train = eval_results["validation_0"]["rmse"]
    rmse_val = eval_results["validation_1"]["rmse"]
    history = pd.DataFrame({
        "epoch": np.arange(1, len(rmse_train) + 1),
        "train_rmse": rmse_train,
        "val_rmse": rmse_val,
    })
    history.to_csv(TBL_DIR / f"training_log{suffix}.csv", index=False)
    print(f"\n[Train] {len(rmse_train)} boosting rounds  ·  best_iter={model.best_iteration}  ·  {train_time:.1f}s")

    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)
    train_metrics = _metrics(y_tr, pred_tr, target_range)
    val_metrics   = _metrics(y_va, pred_va, target_range)
    test_metrics  = _metrics(y_te, pred_te, target_range)

    print("\n[Eval]")
    for name, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(f"  {name:5s}  R²={m['r2']:.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")

    # ---- Stratified eval (per Stage-EDA recommendation) -----------------
    per_source = stratified_regression_metrics(y_te, pred_te, bundle.test_sources)
    per_chem   = stratified_regression_metrics(y_te, pred_te, bundle.test_chemistries)
    print("\n[Eval] Test metrics per source (top 10 by sample count):")
    for r in per_source[:10]:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.1f} cyc  MAE={r['mae']:.1f} cyc")
        else:
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  (below min_n)")
    print("\n[Eval] Test metrics per chemistry:")
    for r in per_chem:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.1f} cyc  MAE={r['mae']:.1f} cyc")
        else:
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  (below min_n)")
    pd.DataFrame(per_source).to_csv(TBL_DIR / f"test_metrics_per_source{suffix}.csv", index=False)
    pd.DataFrame(per_chem).to_csv(TBL_DIR / f"test_metrics_per_chemistry{suffix}.csv", index=False)

    # ---- Censoring-stratified test eval (per Iter-3 §3.11.5 RUL diagnostic) ---
    # Always reported regardless of training mode so all runs are directly comparable.
    print("\n[Eval] Censoring-stratified test metrics:")
    df_full = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "cycling" / "unified.parquet"
    )
    censoring = compute_battery_censoring(df_full)
    test_censored_mask = np.array(
        [censoring.get(bid, False) for bid in bundle.test_battery_ids.values]
    )
    censoring_stratified = {}
    for label, mask in [("uncensored_only", ~test_censored_mask),
                        ("censored_only",   test_censored_mask)]:
        if mask.sum() > 0:
            m = _metrics(y_te[mask], pred_te[mask], target_range)
            censoring_stratified[label] = {"n": int(mask.sum()), **m}
            print(f"  {label:18s}  n={int(mask.sum()):>7,d}  "
                  f"R²={m['r2']:.4f}  "
                  f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
                  f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")

    gates = {
        "rmse_pct_of_range < 2% (full test)":
            test_metrics["rmse_pct_of_range"] < TARGETS["rul_rmse"],
        "rmse_pct_of_range < 2% (uncensored test)":
            censoring_stratified.get("uncensored_only", {}).get("rmse_pct_of_range", 1e9)
            < TARGETS["rul_rmse"],
    }
    print("\n[Gates] " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()))

    metrics_payload = {
        "smoke": args.smoke,
        "tuned": args.tune,
        "audited": args.exclude_capacity,
        "uncensored_train": args.exclude_censored,
        "imputed_rul": args.use_imputed_rul,
        "train_time_s": round(train_time, 2),
        "best_iteration": int(model.best_iteration),
        "n_features": len(bundle.feature_names),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "target_range_cycles": target_range,
        "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in params.items()},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_censoring_stratified": censoring_stratified,
        "gates": gates,
    }
    metrics_path = TBL_DIR / f"metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[Save] metrics → {metrics_path.relative_to(PROJECT_ROOT)}")

    model_path = OUT_DIR / f"xgboost_rul{suffix}.json"
    model.save_model(str(model_path))
    joblib.dump(bundle.scaler, OUT_DIR / f"feature_scaler{suffix}.pkl")
    with open(OUT_DIR / f"feature_names{suffix}.json", "w") as f:
        json.dump(bundle.feature_names, f, indent=2)
    print(f"[Save] model  → {model_path.relative_to(PROJECT_ROOT)}")

    title_tag = " (audited)" if args.exclude_capacity else ""
    plot_loss_curves(history, out_path=FIG_DIR / f"loss_rmse{suffix}.png",
                     title=f"XGBoost RUL{title_tag} — RMSE per boosting round",
                     metric_name="rmse")
    plot_overfit_check(history.rename(columns={"train_rmse": "train_loss",
                                                "val_rmse": "val_loss"}),
                       out_path=FIG_DIR / f"overfit_check{suffix}.png",
                       title=f"XGBoost RUL{title_tag} — overfit/underfit diagnostic")
    metric_text = (f"R²={test_metrics['r2']:.3f}\n"
                   f"RMSE={test_metrics['rmse_cycles']:.1f} cyc\n"
                   f"({test_metrics['rmse_pct_of_range']:.2f}% of range)")
    plot_predicted_vs_actual(y_te, pred_te,
                             out_path=FIG_DIR / f"predicted_vs_actual_test{suffix}.png",
                             title=f"XGBoost RUL{title_tag} — Test set",
                             units="(cycles)", metric_text=metric_text)
    plot_residuals(y_te, pred_te,
                   out_path=FIG_DIR / f"residuals_test{suffix}.png",
                   title=f"XGBoost RUL{title_tag} — Test residuals",
                   units="(cycles)")
    plot_feature_importance(bundle.feature_names,
                            model.feature_importances_,
                            out_path=FIG_DIR / f"feature_importance{suffix}.png",
                            title=f"XGBoost RUL{title_tag} — Built-in feature importance")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")

    if not args.smoke:
        _maybe_run_shap(model, X_te, bundle.feature_names, args.shap_samples, suffix=suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
