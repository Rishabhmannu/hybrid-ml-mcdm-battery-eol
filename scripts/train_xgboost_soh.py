"""
Stage 9 — Train XGBoost SoH regressor on unified.parquet.

Targets per [src/utils/config.py::TARGETS](../src/utils/config.py): R² > 0.95,
RMSE < 2 % SoH, MAE < 1.5 % SoH.

Usage
-----
    python scripts/train_xgboost_soh.py --smoke               # 60 sec sanity run
    python scripts/train_xgboost_soh.py                       # full training
    python scripts/train_xgboost_soh.py --tune --trials 50    # full + Optuna
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

from src.data.training_data import load_feature_bundle
from src.utils.config import MODELS_DIR, RESULTS_DIR, TARGETS, XGBOOST_CONFIG, RANDOM_SEED
from src.utils.metrics import stratified_regression_metrics
from src.utils.plots import (
    plot_feature_importance,
    plot_loss_curves,
    plot_overfit_check,
    plot_predicted_vs_actual,
    plot_residuals,
)

OUT_DIR = MODELS_DIR / "xgboost_soh"
FIG_DIR = RESULTS_DIR / "figures" / "xgboost_soh"
TBL_DIR = RESULTS_DIR / "tables" / "xgboost_soh"


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-3))) * 100),
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
    p = argparse.ArgumentParser(description="Train XGBoost SoH regressor")
    p.add_argument("--smoke", action="store_true", help="Smoke run: tiny sample, few trees")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials")
    p.add_argument("--shap-samples", type=int, default=2000, help="SHAP sample size")
    p.add_argument("--include-high-missing", action="store_true",
                   help="Include ir_ohm + temperature features (87-93%% missing — for ablation only)")
    p.add_argument("--exclude-capacity", action="store_true",
                   help="Iter-3 audited mode: drop capacity_Ah and 5 capacity-derived "
                        "rolling/Δ features. Realistic deployment scenario where capacity "
                        "is the *target* of estimation, not an input.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"XGBoost SoH training  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    bundle = load_feature_bundle(smoke=args.smoke,
                                 include_high_missing=args.include_high_missing,
                                 exclude_capacity_features=args.exclude_capacity)
    suffix = "_audited" if args.exclude_capacity else ""

    X_tr, y_tr = bundle.X_train, bundle.y_train_soh
    X_va, y_va = bundle.X_val, bundle.y_val_soh
    X_te, y_te = bundle.X_test, bundle.y_test_soh

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
    train_metrics = _metrics(y_tr, pred_tr)
    val_metrics   = _metrics(y_va, pred_va)
    test_metrics  = _metrics(y_te, pred_te)

    print("\n[Eval]")
    for name, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(f"  {name:5s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}%  "
              f"MAE={m['mae']:.3f}%  MAPE={m['mape']:.2f}%")

    # ---- Stratified eval (per Stage-EDA recommendation) -----------------
    per_source = stratified_regression_metrics(y_te, pred_te, bundle.test_sources)
    per_chem   = stratified_regression_metrics(y_te, pred_te, bundle.test_chemistries)
    print("\n[Eval] Test metrics per source (top 10 by sample count):")
    for r in per_source[:10]:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.3f}%  MAE={r['mae']:.3f}%")
        else:
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  (below min_n)")
    print("\n[Eval] Test metrics per chemistry:")
    for r in per_chem:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.3f}%  MAE={r['mae']:.3f}%")
        else:
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  (below min_n)")
    pd.DataFrame(per_source).to_csv(TBL_DIR / f"test_metrics_per_source{suffix}.csv", index=False)
    pd.DataFrame(per_chem).to_csv(TBL_DIR / f"test_metrics_per_chemistry{suffix}.csv", index=False)

    targets_check = {
        "r2 > 0.95": test_metrics["r2"] > TARGETS["soh_r2"],
        "rmse < 2%": test_metrics["rmse"] < TARGETS["soh_rmse"],
        "mae < 1.5%": test_metrics["mae"] < TARGETS["soh_mae"],
    }
    print("\n[Gates] " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in targets_check.items()))

    metrics_payload = {
        "smoke": args.smoke,
        "tuned": args.tune,
        "train_time_s": round(train_time, 2),
        "best_iteration": int(model.best_iteration),
        "n_features": len(bundle.feature_names),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in params.items()},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "gates": targets_check,
    }
    metrics_path = TBL_DIR / f"metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[Save] metrics → {metrics_path.relative_to(PROJECT_ROOT)}")

    model_path = OUT_DIR / f"xgboost_soh{suffix}.json"
    model.save_model(str(model_path))
    joblib.dump(bundle.scaler, OUT_DIR / f"feature_scaler{suffix}.pkl")
    with open(OUT_DIR / f"feature_names{suffix}.json", "w") as f:
        json.dump(bundle.feature_names, f, indent=2)
    print(f"[Save] model  → {model_path.relative_to(PROJECT_ROOT)}")

    title_tag = " (audited)" if args.exclude_capacity else ""
    plot_loss_curves(history, out_path=FIG_DIR / f"loss_rmse{suffix}.png",
                     title=f"XGBoost SoH{title_tag} — RMSE per boosting round",
                     metric_name="rmse")
    plot_overfit_check(history.rename(columns={"train_rmse": "train_loss",
                                                "val_rmse": "val_loss"}),
                       out_path=FIG_DIR / f"overfit_check{suffix}.png",
                       title=f"XGBoost SoH{title_tag} — overfit/underfit diagnostic")
    metric_text = (f"R²={test_metrics['r2']:.3f}\n"
                   f"RMSE={test_metrics['rmse']:.2f}%\n"
                   f"MAE={test_metrics['mae']:.2f}%")
    plot_predicted_vs_actual(y_te, pred_te,
                             out_path=FIG_DIR / f"predicted_vs_actual_test{suffix}.png",
                             title=f"XGBoost SoH{title_tag} — Test set",
                             units="(SoH %)", metric_text=metric_text)
    plot_residuals(y_te, pred_te,
                   out_path=FIG_DIR / f"residuals_test{suffix}.png",
                   title=f"XGBoost SoH{title_tag} — Test residuals",
                   units="(SoH %)")
    plot_feature_importance(bundle.feature_names,
                            model.feature_importances_,
                            out_path=FIG_DIR / f"feature_importance{suffix}.png",
                            title=f"XGBoost SoH{title_tag} — Built-in feature importance")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")

    if not args.smoke:
        _maybe_run_shap(model, X_te, bundle.feature_names, args.shap_samples, suffix=suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
