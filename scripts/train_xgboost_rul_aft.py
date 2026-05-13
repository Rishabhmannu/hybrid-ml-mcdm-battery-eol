"""
Iter-3 §3.11.5 Hypothesis A intervention — XGBoost RUL with survival:aft.

Replaces the standard regression objective with **Accelerated Failure Time
survival regression** (`objective="survival:aft"`) so right-censored batteries
contribute "this cell lasted AT LEAST N cycles" instead of having their RUL
labels fabricated as `max(observed_cycle) − current_cycle`.

Per Iter-3 §3.11.5 RUL diagnostic:
- 18.9 % of corpus is right-censored
- On the audited regression model, censored-cell test RMSE is 3.7× the
  uncensored-cell RMSE (7.77 % vs 2.10 % of range)
- Experiment 2 (drop censored from train) confirmed labels are the bottleneck
  but dropped distributional coverage

AFT label encoding:
- Uncensored row at cycle N (true EoL observed at cycle N+R): lower=R, upper=R
- Censored row at cycle N (last observed at cycle N+R): lower=R, upper=+∞
- AFT predicts log-RUL; we exponentiate for cycle-space comparison vs LSTM /
  baseline XGBoost RUL.

Usage
-----
    python scripts/train_xgboost_rul_aft.py --smoke
    python scripts/train_xgboost_rul_aft.py --exclude-capacity
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
from src.utils.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TARGETS, XGBOOST_CONFIG
from src.utils.metrics import stratified_regression_metrics
from src.utils.plots import (
    plot_feature_importance,
    plot_loss_curves,
    plot_overfit_check,
    plot_predicted_vs_actual,
    plot_residuals,
)

OUT_DIR = MODELS_DIR / "xgboost_rul_aft"
FIG_DIR = RESULTS_DIR / "figures" / "xgboost_rul_aft"
TBL_DIR = RESULTS_DIR / "tables" / "xgboost_rul_aft"


# =============================================================================
# AFT label construction
# =============================================================================

def build_aft_labels(rul: np.ndarray,
                     battery_ids: pd.Series,
                     censoring_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build (lower, upper) label arrays for XGBoost AFT.

    For uncensored cells: lower = upper = rul (exact event time).
    For right-censored cells: lower = rul (lasted at least this long),
    upper = +inf (true RUL unknown).

    AFT requires strictly positive labels, so we add a small epsilon when
    rul == 0 (which can happen at the cycle equal to or after observed EoL).
    """
    censored = np.array(
        [censoring_map.get(bid, False) for bid in battery_ids.values],
        dtype=bool,
    )
    eps = 1.0  # one cycle floor — strictly positive AFT requirement
    lower = np.maximum(rul, eps).astype(np.float32)
    upper = np.where(censored, np.inf, lower).astype(np.float32)
    return lower, upper, censored


# =============================================================================
# Metrics — AFT predictions are in log-RUL space; exponentiate for compare
# =============================================================================

def _metrics(y_true_cycles, y_pred_cycles, target_range: float) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true_cycles, y_pred_cycles)))
    mae = float(mean_absolute_error(y_true_cycles, y_pred_cycles))
    r2 = float(r2_score(y_true_cycles, y_pred_cycles)) if len(np.unique(y_true_cycles)) > 1 else float("nan")
    return {
        "r2": r2,
        "rmse_cycles": rmse,
        "mae_cycles": mae,
        "rmse_pct_of_range": rmse / max(target_range, 1.0) * 100.0,
        "mae_pct_of_range": mae / max(target_range, 1.0) * 100.0,
    }


def _predict_cycles(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """AFT booster.predict() returns survival time in cycles directly when
    output_margin=False (default) — XGBoost handles the exp() internally.
    See https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html"""
    return booster.predict(xgb.DMatrix(X))


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Train XGBoost RUL with AFT survival objective")
    p.add_argument("--smoke", action="store_true", help="Smoke run: tiny sample, few trees")
    p.add_argument("--exclude-capacity", action="store_true",
                   help="Iter-3 audited mode: drop capacity_Ah and 5 capacity-derived "
                        "rolling/Δ features. Matches the audited SoH model's feature set.")
    p.add_argument("--include-high-missing", action="store_true",
                   help="Include ir_ohm + temperature features (87-93%% missing — for ablation only)")
    p.add_argument("--aft-distribution", choices=["normal", "logistic", "extreme"],
                   default="normal",
                   help="AFT distribution family (XGBoost default: normal). "
                        "Normal is appropriate when log-RUL is approximately Gaussian; "
                        "logistic is heavier-tailed; extreme is Weibull-like.")
    p.add_argument("--aft-sigma", type=float, default=1.0,
                   help="AFT scale parameter σ. XGBoost will tune internally during "
                        "boosting; this is the initial value.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"XGBoost RUL — AFT survival regression  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    bundle = load_feature_bundle(smoke=args.smoke,
                                 include_high_missing=args.include_high_missing,
                                 exclude_capacity_features=args.exclude_capacity,
                                 exclude_censored_batteries=False)  # AFT uses ALL data
    suffix = "_audited" if args.exclude_capacity else ""

    X_tr, X_va, X_te = bundle.X_train, bundle.X_val, bundle.X_test
    y_tr_cyc = bundle.y_train_rul
    y_va_cyc = bundle.y_val_rul
    y_te_cyc = bundle.y_test_rul

    # Need raw df once to compute battery-level censoring map
    print("\n[Censoring] Computing battery-level censoring map for AFT labels ...")
    df_full = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
    )
    censoring_map = compute_battery_censoring(df_full)
    n_censored = sum(1 for v in censoring_map.values() if v)
    n_total = len(censoring_map)
    print(f"  {n_censored:,}/{n_total:,} batteries censored ({n_censored/n_total*100:.1f}%)")

    y_tr_lo, y_tr_hi, c_tr = build_aft_labels(y_tr_cyc, bundle.train_battery_ids, censoring_map)
    y_va_lo, y_va_hi, c_va = build_aft_labels(y_va_cyc, bundle.val_battery_ids,   censoring_map)
    y_te_lo, y_te_hi, c_te = build_aft_labels(y_te_cyc, bundle.test_battery_ids,  censoring_map)
    print(f"[AFT labels] train: {(~c_tr).sum():,} uncensored + {c_tr.sum():,} censored "
          f"(rate={c_tr.mean()*100:.1f}%)")
    print(f"[AFT labels] val:   {(~c_va).sum():,} uncensored + {c_va.sum():,} censored "
          f"(rate={c_va.mean()*100:.1f}%)")
    print(f"[AFT labels] test:  {(~c_te).sum():,} uncensored + {c_te.sum():,} censored "
          f"(rate={c_te.mean()*100:.1f}%)")

    target_range = float(np.ptp(y_tr_cyc)) if len(y_tr_cyc) > 0 else 1.0
    print(f"[Data] RUL range (train) = {target_range:.0f} cycles")

    # AFT requires the DMatrix label_lower_bound / label_upper_bound API.
    dtr = xgb.DMatrix(X_tr); dtr.set_float_info("label_lower_bound", y_tr_lo); dtr.set_float_info("label_upper_bound", y_tr_hi)
    dva = xgb.DMatrix(X_va); dva.set_float_info("label_lower_bound", y_va_lo); dva.set_float_info("label_upper_bound", y_va_hi)
    dte = xgb.DMatrix(X_te); dte.set_float_info("label_lower_bound", y_te_lo); dte.set_float_info("label_upper_bound", y_te_hi)

    base = {**XGBOOST_CONFIG}
    n_estimators = base.pop("n_estimators", 500)
    if args.smoke:
        n_estimators = 50
        base["max_depth"] = 4
    params = {
        **base,
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": args.aft_distribution,
        "aft_loss_distribution_scale": args.aft_sigma,
        # XGBoost AFT only supports tree_method=hist or approx (not exact).
        "tree_method": "hist",
        "verbosity": 0,
    }
    print(f"\n[Train] params: {json.dumps(params, indent=2)}")
    print(f"[Train] num_boost_round={n_estimators}  early_stopping=30")

    evals = [(dtr, "train"), (dva, "val")]
    evals_result: dict = {}
    t0 = time.time()
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    train_time = time.time() - t0
    best_iter = booster.best_iteration
    print(f"\n[Train] {len(evals_result['train']['aft-nloglik'])} rounds  ·  "
          f"best_iter={best_iter}  ·  {train_time:.1f}s")

    # Loss history
    history = pd.DataFrame({
        "epoch": np.arange(1, len(evals_result["train"]["aft-nloglik"]) + 1),
        "train_aft_nloglik": evals_result["train"]["aft-nloglik"],
        "val_aft_nloglik":   evals_result["val"]["aft-nloglik"],
    })
    history.to_csv(TBL_DIR / f"training_log{suffix}.csv", index=False)

    pred_tr = _predict_cycles(booster, X_tr)
    pred_va = _predict_cycles(booster, X_va)
    pred_te = _predict_cycles(booster, X_te)

    # Metrics on UNCENSORED cells only (where ground-truth RUL is known).
    # On censored cells the cycle-space "true RUL" is fabricated, so RMSE there
    # is not a meaningful comparison number — we still report it for transparency.
    train_metrics_unc = _metrics(y_tr_cyc[~c_tr], pred_tr[~c_tr], target_range)
    val_metrics_unc   = _metrics(y_va_cyc[~c_va], pred_va[~c_va], target_range)
    test_metrics_unc  = _metrics(y_te_cyc[~c_te], pred_te[~c_te], target_range)

    # Headline (full-test) metric — for transparency vs LSTM / baseline XGB-RUL.
    # Note: this number compares against fabricated labels on censored cells,
    # so it is NOT the gate-relevant number; uncensored-test is.
    train_metrics_full = _metrics(y_tr_cyc, pred_tr, target_range)
    val_metrics_full   = _metrics(y_va_cyc, pred_va, target_range)
    test_metrics_full  = _metrics(y_te_cyc, pred_te, target_range)

    test_metrics_cen = _metrics(y_te_cyc[c_te], pred_te[c_te], target_range) if c_te.any() else {}

    print("\n[Eval — uncensored only (gate-relevant)]")
    for name, m in [("train", train_metrics_unc), ("val", val_metrics_unc), ("test", test_metrics_unc)]:
        print(f"  {name:5s}  R²={m['r2']:.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")
    print("\n[Eval — full test (transparency, fabricated labels included)]")
    for name, m in [("train", train_metrics_full), ("val", val_metrics_full), ("test", test_metrics_full)]:
        print(f"  {name:5s}  R²={m['r2']:.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc")
    if test_metrics_cen:
        print("\n[Eval — censored test only (fabricated labels — diagnostic only)]")
        m = test_metrics_cen
        print(f"  test   R²={m['r2']:.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc")

    # ---- Stratified test (uncensored subset only — meaningful comparisons) ----
    test_unc_idx = np.where(~c_te)[0]
    per_source = stratified_regression_metrics(
        y_te_cyc[~c_te], pred_te[~c_te],
        bundle.test_sources.iloc[test_unc_idx].reset_index(drop=True),
    )
    per_chem = stratified_regression_metrics(
        y_te_cyc[~c_te], pred_te[~c_te],
        bundle.test_chemistries.iloc[test_unc_idx].reset_index(drop=True),
    )
    print("\n[Eval] Uncensored-test metrics per source (top 10):")
    for r in per_source[:10]:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:30s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.1f} cyc  MAE={r['mae']:.1f} cyc")
    print("\n[Eval] Uncensored-test metrics per chemistry:")
    for r in per_chem:
        if pd.notna(r["r2"]):
            print(f"  {r['stratum']:10s}  n={r['n']:>7,d}  "
                  f"R²={r['r2']:.4f}  RMSE={r['rmse']:.1f} cyc  MAE={r['mae']:.1f} cyc")
    pd.DataFrame(per_source).to_csv(TBL_DIR / f"test_metrics_per_source{suffix}.csv", index=False)
    pd.DataFrame(per_chem).to_csv(TBL_DIR / f"test_metrics_per_chemistry{suffix}.csv", index=False)

    gates = {
        "rmse_pct_of_range < 2% (uncensored test, gate-relevant)":
            test_metrics_unc["rmse_pct_of_range"] < TARGETS["rul_rmse"],
        "rmse_pct_of_range < 2% (full test, transparency)":
            test_metrics_full["rmse_pct_of_range"] < TARGETS["rul_rmse"],
    }
    print("\n[Gates]")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    metrics_payload = {
        "smoke": args.smoke,
        "audited": args.exclude_capacity,
        "objective": "survival:aft",
        "aft_distribution": args.aft_distribution,
        "aft_sigma": args.aft_sigma,
        "train_time_s": round(train_time, 2),
        "best_iteration": int(best_iter),
        "n_features": len(bundle.feature_names),
        "n_train": int(len(y_tr_cyc)),
        "n_train_uncensored": int((~c_tr).sum()),
        "n_train_censored":   int(c_tr.sum()),
        "n_val": int(len(y_va_cyc)),
        "n_test": int(len(y_te_cyc)),
        "target_range_cycles": target_range,
        "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in params.items()},
        "train_uncensored": train_metrics_unc,
        "val_uncensored":   val_metrics_unc,
        "test_uncensored":  test_metrics_unc,
        "train_full": train_metrics_full,
        "val_full":   val_metrics_full,
        "test_full":  test_metrics_full,
        "test_censored": test_metrics_cen,
        "gates": gates,
    }
    metrics_path = TBL_DIR / f"metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[Save] metrics → {metrics_path.relative_to(PROJECT_ROOT)}")

    booster.save_model(str(OUT_DIR / f"xgboost_rul_aft{suffix}.json"))
    joblib.dump(bundle.scaler, OUT_DIR / f"feature_scaler{suffix}.pkl")
    with open(OUT_DIR / f"feature_names{suffix}.json", "w") as f:
        json.dump(bundle.feature_names, f, indent=2)
    print(f"[Save] model  → {(OUT_DIR / f'xgboost_rul_aft{suffix}.json').relative_to(PROJECT_ROOT)}")

    title_tag = " (audited)" if args.exclude_capacity else ""
    plot_loss_curves(history.rename(columns={"train_aft_nloglik": "train_loss",
                                              "val_aft_nloglik": "val_loss"}),
                     out_path=FIG_DIR / f"loss_aft_nloglik{suffix}.png",
                     title=f"XGBoost RUL AFT{title_tag} — neg-log-lik per round",
                     metric_name="loss")
    plot_overfit_check(history.rename(columns={"train_aft_nloglik": "train_loss",
                                                "val_aft_nloglik": "val_loss"}),
                       out_path=FIG_DIR / f"overfit_check{suffix}.png",
                       title=f"XGBoost RUL AFT{title_tag} — overfit/underfit diagnostic")
    metric_text = (f"R²={test_metrics_unc['r2']:.3f}\n"
                   f"RMSE={test_metrics_unc['rmse_cycles']:.1f} cyc\n"
                   f"({test_metrics_unc['rmse_pct_of_range']:.2f}% of range)")
    plot_predicted_vs_actual(y_te_cyc[~c_te], pred_te[~c_te],
                             out_path=FIG_DIR / f"predicted_vs_actual_test_uncensored{suffix}.png",
                             title=f"XGBoost RUL AFT{title_tag} — Uncensored test",
                             units="(cycles)", metric_text=metric_text)
    plot_residuals(y_te_cyc[~c_te], pred_te[~c_te],
                   out_path=FIG_DIR / f"residuals_test_uncensored{suffix}.png",
                   title=f"XGBoost RUL AFT{title_tag} — Uncensored test residuals",
                   units="(cycles)")
    fi_dict = booster.get_score(importance_type="gain")
    fi = np.array([fi_dict.get(f"f{i}", 0.0) for i in range(len(bundle.feature_names))])
    plot_feature_importance(bundle.feature_names, fi,
                            out_path=FIG_DIR / f"feature_importance{suffix}.png",
                            title=f"XGBoost RUL AFT{title_tag} — Built-in feature importance (gain)")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
