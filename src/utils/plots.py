"""
Standardized plotting utilities for Stage 9 model evaluation.

Single professional theme, reused across all training scripts. Each helper
saves PNG (300 dpi) + PDF for paper figures, returns the path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

_PALETTE = {
    "train": "#1F4E79",      # deep blue
    "val":   "#C0392B",      # warm red
    "test":  "#27AE60",      # green
    "ideal": "#7F8C8D",      # gray
    "accent": "#F39C12",     # amber
    "shade": "#D6EAF8",      # pale blue fill
}


def apply_theme():
    """Set matplotlib rcParams for a consistent paper-ready theme."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "grid.color": "#B0B0B0",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.prop_cycle": plt.cycler(
            color=[_PALETTE["train"], _PALETTE["val"], _PALETTE["test"],
                   _PALETTE["accent"], "#8E44AD", "#16A085"]
        ),
    })


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _save(fig, out_path: Path, also_pdf: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Training-curve plots
# --------------------------------------------------------------------------

def plot_loss_curves(
    history: pd.DataFrame,
    *,
    out_path: Path,
    title: str = "Training & Validation Loss",
    metric_name: str = "loss",
    log_y: bool = False,
):
    """`history` columns: epoch | train_<metric_name> | val_<metric_name>."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    train_col = f"train_{metric_name}"
    val_col = f"val_{metric_name}"

    if train_col in history.columns:
        ax.plot(history["epoch"], history[train_col],
                marker="o", markersize=3, label=f"Train {metric_name}",
                color=_PALETTE["train"])
    if val_col in history.columns:
        ax.plot(history["epoch"], history[val_col],
                marker="s", markersize=3, label=f"Val {metric_name}",
                color=_PALETTE["val"])

    if val_col in history.columns and len(history) > 0:
        best_idx = history[val_col].idxmin()
        ax.axvline(history.loc[best_idx, "epoch"], color="#7F8C8D",
                   linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Best epoch ({int(history.loc[best_idx, 'epoch'])})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="upper right")
    return _save(fig, out_path, also_pdf=True)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: Path,
    title: str = "Predicted vs Actual",
    units: str = "",
    metric_text: str | None = None,
):
    """Scatter of predicted vs actual with the y=x ideal line and a residual band."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=_PALETTE["train"], edgecolor="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=_PALETTE["ideal"], linestyle="--",
            linewidth=1, label="y = x (ideal)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Actual {units}".strip())
    ax.set_ylabel(f"Predicted {units}".strip())
    ax.set_title(title)

    if metric_text:
        ax.text(0.04, 0.96, metric_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#B0B0B0",
                          boxstyle="round,pad=0.4", alpha=0.9))
    ax.legend(loc="lower right")
    return _save(fig, out_path, also_pdf=True)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: Path,
    title: str = "Residuals vs Predicted",
    units: str = "",
):
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    residuals = y_pred - y_true

    axes[0].scatter(y_pred, residuals, s=8, alpha=0.3,
                    color=_PALETTE["train"], edgecolor="none")
    axes[0].axhline(0, color=_PALETTE["ideal"], linestyle="--", linewidth=1)
    axes[0].set_xlabel(f"Predicted {units}".strip())
    axes[0].set_ylabel("Residual (pred − actual)")
    axes[0].set_title(title)

    axes[1].hist(residuals, bins=50, color=_PALETTE["train"], alpha=0.85)
    axes[1].axvline(0, color=_PALETTE["ideal"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    txt = (f"mean={residuals.mean():.3f}\n"
           f"std={residuals.std():.3f}\n"
           f"|max|={np.abs(residuals).max():.3f}")
    axes[1].text(0.96, 0.95, txt, transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="#B0B0B0",
                           boxstyle="round,pad=0.4", alpha=0.9))
    return _save(fig, out_path, also_pdf=True)


def plot_overfit_check(
    history: pd.DataFrame,
    *,
    out_path: Path,
    title: str = "Overfit / Underfit Diagnostic",
    metric_name: str = "loss",
):
    """Visualizes train vs val gap; shaded zones flag over/underfit regimes."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    train_col = f"train_{metric_name}"
    val_col = f"val_{metric_name}"

    train = history[train_col].to_numpy()
    val = history[val_col].to_numpy()
    gap = val - train
    epochs = history["epoch"].to_numpy()

    ax.plot(epochs, train, label="train", color=_PALETTE["train"], marker="o", markersize=2.5)
    ax.plot(epochs, val,   label="val",   color=_PALETTE["val"],   marker="s", markersize=2.5)
    ax.fill_between(epochs, train, val, where=(val > train),
                    color=_PALETTE["val"], alpha=0.10, label="overfit gap")

    last_gap = gap[-1] if len(gap) else float("nan")
    last_train = train[-1] if len(train) else float("nan")
    if last_train > 0:
        rel_gap = last_gap / max(last_train, 1e-6)
    else:
        rel_gap = float("nan")
    if rel_gap > 0.20:
        verdict = f"likely overfitting (val/train gap = {rel_gap*100:.1f}%)"
    elif rel_gap < -0.05:
        verdict = f"val < train — small dataset / regularization heavy ({rel_gap*100:.1f}%)"
    elif last_train > 0 and last_train > val.min():
        verdict = "underfitting suspected (train still high)"
    else:
        verdict = f"good fit (val/train gap = {rel_gap*100:.1f}%)"

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{title} — {verdict}")
    ax.legend(loc="upper right")
    return _save(fig, out_path, also_pdf=True)


# --------------------------------------------------------------------------
# Classification + diagnostic plots
# --------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Iterable, y_pred: Iterable, *,
    labels: list,
    out_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
):
    apply_theme()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            color = "white" if v > cm.max() * 0.55 else "#1A1A1A"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045)
    return _save(fig, out_path, also_pdf=True)


def plot_feature_importance(
    feature_names: list, importances: np.ndarray, *,
    out_path: Path,
    title: str = "Feature Importance",
    top_n: int = 20,
):
    apply_theme()
    df = (pd.DataFrame({"feature": feature_names, "importance": importances})
          .sort_values("importance", ascending=True)
          .tail(top_n))
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.28 * len(df))))
    ax.barh(df["feature"], df["importance"], color=_PALETTE["train"], alpha=0.9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return _save(fig, out_path, also_pdf=True)


def plot_anomaly_scores(
    scores: np.ndarray, threshold: float,
    *,
    out_path: Path,
    title: str = "Anomaly Score Distribution",
):
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(scores, bins=80, color=_PALETTE["train"], alpha=0.85)
    ax.axvline(threshold, color=_PALETTE["val"], linestyle="--",
               label=f"threshold={threshold:.4f}")
    n_anom = int((scores > threshold).sum())
    ax.set_xlabel("Anomaly score (reconstruction error)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  ·  flagged: {n_anom} ({n_anom/max(len(scores),1)*100:.2f}%)")
    ax.legend()
    return _save(fig, out_path, also_pdf=True)
