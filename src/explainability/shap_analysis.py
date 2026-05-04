"""
SHAP (SHapley Additive exPlanations) analysis for XGBoost SoH model.
Provides global feature importance and per-battery prediction explanations.

References:
- BOA-XGBoost + TreeSHAP (2024), MDPI Batteries 10(11), 394
- Multi-source fusion with SHAP (2025), arXiv:2504.18230
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import FIGURES_DIR


def create_explainer(model):
    """Create TreeSHAP explainer for tree-based models (XGBoost/LightGBM)."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X):
    """Compute SHAP values for a dataset."""
    return explainer.shap_values(X)


def plot_summary_bar(shap_values, X, save_path=None):
    """Global feature importance bar plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_summary_beeswarm(shap_values, X, save_path=None):
    """Detailed beeswarm plot showing feature value impact direction."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Impact (Beeswarm)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_force_single(explainer, shap_values, X, idx=0, save_path=None):
    """Force plot for a single battery prediction."""
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx] if hasattr(X, "iloc") else X[idx],
        matplotlib=True,
        show=False,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_dependence(shap_values, X, feature_name, save_path=None):
    """Dependence plot for a specific feature."""
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature_name, shap_values, X, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def get_top_features(shap_values, feature_names, top_n=10):
    """Get top N most important features by mean absolute SHAP value."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    return importance_df.head(top_n)


def run_full_shap_analysis(model, X_test, feature_names=None, output_dir=None):
    """Run complete SHAP analysis and save all plots."""
    output_dir = Path(output_dir or FIGURES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running SHAP analysis...")

    # Create explainer and compute SHAP values
    explainer = create_explainer(model)
    shap_values = compute_shap_values(explainer, X_test)

    # Generate all plots
    plot_summary_bar(shap_values, X_test, output_dir / "shap_summary_bar.png")
    plot_summary_beeswarm(shap_values, X_test, output_dir / "shap_beeswarm.png")

    # Top features
    names = feature_names if feature_names is not None else (
        X_test.columns.tolist() if hasattr(X_test, "columns") else
        [f"f{i}" for i in range(X_test.shape[1])]
    )
    top_features = get_top_features(shap_values, names)
    print("\nTop 10 Features (by mean |SHAP|):")
    print(top_features.to_string(index=False))

    # Save top features table
    top_features.to_csv(output_dir.parent / "tables" / "shap_top_features.csv", index=False)

    # Dependence plots for top 3 features
    for feat in top_features["feature"].head(3):
        if feat in names:
            plot_dependence(
                shap_values, X_test, feat,
                output_dir / f"shap_dependence_{feat}.png"
            )

    return shap_values, explainer, top_features
