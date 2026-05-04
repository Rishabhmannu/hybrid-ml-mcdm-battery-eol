"""
XGBoost model for State-of-Health (SoH) regression.
Predicts battery SoH from engineered cycling features.

References:
- Hybrid XGBoost-LSTM (2025), ScienceDirect — R2=0.983
- BOA-XGBoost + TreeSHAP (2024), MDPI Batteries
- Rout et al. (2025), Scientific Reports — XGBoost achieves 0.9402
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import XGBOOST_CONFIG, OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, RANDOM_SEED, MODELS_DIR


def create_xgboost_model(params: dict = None) -> xgb.XGBRegressor:
    """Create XGBoost regressor with default or custom parameters."""
    config = {**XGBOOST_CONFIG, **(params or {})}
    return xgb.XGBRegressor(**config)


def optuna_objective(trial, X_train, y_train):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
    }

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=OPTUNA_CV_FOLDS, scoring="r2", n_jobs=-1
    )
    return scores.mean()


def tune_hyperparameters(X_train, y_train, n_trials: int = None):
    """Run Optuna hyperparameter optimization."""
    n_trials = n_trials or OPTUNA_N_TRIALS

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"Best R2 (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save study
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, MODELS_DIR / "optuna_study_xgboost.pkl")

    return study.best_params


def train_and_evaluate(X_train, y_train, X_test, y_test, params: dict = None):
    """Train XGBoost and return model + metrics."""
    model = create_xgboost_model(params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
    }

    print(f"XGBoost SoH Results:")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "xgboost_soh.json"))

    return model, metrics
