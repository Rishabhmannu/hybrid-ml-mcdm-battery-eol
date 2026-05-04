"""
Grade classification: maps SoH predictions to EoL route grades (A/B/C/D).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import GRADE_THRESHOLDS, GRADE_ROUTES


def classify_battery(soh: float, rul: float = None) -> dict:
    """
    Classify a single battery into a grade based on SoH.

    Returns dict with grade, route recommendation, and confidence context.
    """
    if soh > GRADE_THRESHOLDS["A"]:
        grade = "A"
    elif soh > GRADE_THRESHOLDS["B"]:
        grade = "B"
    elif soh > GRADE_THRESHOLDS["C"]:
        grade = "C"
    else:
        grade = "D"

    return {
        "grade": grade,
        "soh": soh,
        "rul": rul,
        "recommended_route": GRADE_ROUTES[grade],
    }


def classify_batch(soh_predictions: np.ndarray, rul_predictions: np.ndarray = None) -> pd.DataFrame:
    """Classify a batch of batteries."""
    results = []
    rul_preds = rul_predictions if rul_predictions is not None else [None] * len(soh_predictions)

    for soh, rul in zip(soh_predictions, rul_preds):
        results.append(classify_battery(soh, rul))

    return pd.DataFrame(results)


def evaluate_classification(y_true_grades: np.ndarray, y_pred_grades: np.ndarray) -> dict:
    """Evaluate grade classification performance."""
    accuracy = accuracy_score(y_true_grades, y_pred_grades)
    f1_macro = f1_score(y_true_grades, y_pred_grades, average="macro")
    f1_per_class = f1_score(y_true_grades, y_pred_grades, average=None,
                            labels=["A", "B", "C", "D"])
    cm = confusion_matrix(y_true_grades, y_pred_grades, labels=["A", "B", "C", "D"])
    report = classification_report(y_true_grades, y_pred_grades, labels=["A", "B", "C", "D"])

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(["A", "B", "C", "D"], f1_per_class)),
        "confusion_matrix": cm,
        "report": report,
    }

    print(f"Grade Classification Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 per class: A={f1_per_class[0]:.3f}, B={f1_per_class[1]:.3f}, "
          f"C={f1_per_class[2]:.3f}, D={f1_per_class[3]:.3f}")
    print(f"\n{report}")

    return metrics
