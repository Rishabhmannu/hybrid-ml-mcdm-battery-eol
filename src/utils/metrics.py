"""
Evaluation metric functions for all model components.
"""
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)


def regression_metrics(y_true, y_pred):
    """Compute all regression metrics for SoH/RUL evaluation."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }


def classification_metrics(y_true, y_pred, labels=None):
    """Compute classification metrics for grade evaluation."""
    labels = labels or ["A", "B", "C", "D"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "f1_per_class": dict(zip(
            labels,
            f1_score(y_true, y_pred, average=None, labels=labels)
        )),
    }


def stratified_regression_metrics(
    y_true,
    y_pred,
    strata,
    *,
    min_n: int = 20,
):
    """Compute regression metrics per unique value in `strata`.

    Returns a list of dicts (`stratum`, `n`, `r2`, `rmse`, `mae`, `mape`),
    one per stratum, sorted by sample count (descending). Strata with
    fewer than `min_n` rows are reported with `r2 = NaN` to avoid
    misleading single-cell metrics.
    """
    import pandas as pd

    df = pd.DataFrame({"y_true": np.asarray(y_true).ravel(),
                       "y_pred": np.asarray(y_pred).ravel(),
                       "stratum": np.asarray(strata)})
    rows = []
    for stratum, sub in df.groupby("stratum", sort=False):
        n = len(sub)
        if n < min_n:
            rows.append({"stratum": stratum, "n": n, "r2": float("nan"),
                         "rmse": float("nan"), "mae": float("nan"),
                         "mape": float("nan")})
            continue
        m = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append({"stratum": stratum, "n": n, **m})
    return sorted(rows, key=lambda r: -r["n"])


def stratified_classification_metrics(
    y_true,
    y_pred,
    strata,
    *,
    labels=None,
    min_n: int = 20,
):
    """Compute accuracy + F1-macro per unique value in `strata`."""
    import pandas as pd

    labels = labels or ["A", "B", "C", "D"]
    df = pd.DataFrame({"y_true": np.asarray(y_true).ravel(),
                       "y_pred": np.asarray(y_pred).ravel(),
                       "stratum": np.asarray(strata)})
    rows = []
    for stratum, sub in df.groupby("stratum", sort=False):
        n = len(sub)
        if n < min_n:
            rows.append({"stratum": stratum, "n": n,
                         "accuracy": float("nan"), "f1_macro": float("nan")})
            continue
        rows.append({"stratum": stratum, "n": n,
                     "accuracy": float(accuracy_score(sub["y_true"], sub["y_pred"])),
                     "f1_macro": float(f1_score(sub["y_true"], sub["y_pred"],
                                                 average="macro", labels=labels,
                                                 zero_division=0))})
    return sorted(rows, key=lambda r: -r["n"])


def check_targets(metrics: dict, targets: dict) -> dict:
    """Check if metrics meet quality gate targets."""
    results = {}
    for metric, target in targets.items():
        if metric in metrics:
            actual = metrics[metric]
            if metric in ["rmse", "mae", "mape"]:
                passed = actual <= target
            else:
                passed = actual >= target
            results[metric] = {
                "actual": actual,
                "target": target,
                "passed": passed,
            }
    return results
